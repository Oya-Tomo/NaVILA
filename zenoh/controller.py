"""Thread-safe control state machine for the NaVILA Zenoh node."""

from __future__ import annotations

import math
import re
import time
from collections import deque
from dataclasses import dataclass
from threading import RLock
from typing import Callable, Literal, Sequence

from config import (
    ActionState,
    ActionType,
    Go2NodeState,
    HeartbeatCommand,
    InferenceState,
    InstructionCommand,
    Lifecycle,
    NodeConfig,
    NodeState,
    ParsedAction,
    StartCommand,
    StopCommand,
    VelocityCommand,
)
from PIL import Image

ZERO_VELOCITY = VelocityCommand(vx=0.0, vy=0.0, vyaw=0.0)
FORWARD_AMOUNTS_CM = (25, 50, 75)
TURN_AMOUNTS_DEGREES = (15, 30, 45)

_ACTION_PATTERNS = {
    ActionType.STOP: re.compile(r"\bstop\b", re.IGNORECASE),
    ActionType.FORWARD: re.compile(r"\b(?:is\s+)?move forward\b", re.IGNORECASE),
    ActionType.LEFT: re.compile(r"\b(?:is\s+)?turn left\b", re.IGNORECASE),
    ActionType.RIGHT: re.compile(r"\b(?:is\s+)?turn right\b", re.IGNORECASE),
}
_AMOUNT_SUFFIX_PATTERNS = {
    ActionType.FORWARD: re.compile(r"\s+(\d+)\s*cm\b", re.IGNORECASE),
    ActionType.LEFT: re.compile(r"\s+(\d+)\s*degrees?\b", re.IGNORECASE),
    ActionType.RIGHT: re.compile(r"\s+(\d+)\s*degrees?\b", re.IGNORECASE),
}
_PUNCTUATION_ONLY = re.compile(r"^[\s.!?,;:]*$")
_UNSUPPORTED_ACTION = re.compile(r"\b(?:move backward|move (?:left|right))\b", re.IGNORECASE)


class ActionParseError(ValueError):
    """Raised when model output cannot be mapped to one safe action."""


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    instruction: str
    frames: tuple[Image.Image, ...]


@dataclass(frozen=True, slots=True)
class ControlEffects:
    zero_velocity: bool = False
    posture: Literal["stand", "down"] | None = None


def _nearest(value: int, choices: Sequence[int]) -> int:
    return min(choices, key=lambda candidate: (abs(candidate - value), candidate))


def parse_navigation_action(output: str, config: NodeConfig) -> ParsedAction:
    """Parse one eval-compatible output into a bounded velocity and duration."""

    if _UNSUPPORTED_ACTION.search(output):
        raise ActionParseError(f"unsupported action found in model output: {output!r}")

    matches = [
        (action, match) for action, pattern in _ACTION_PATTERNS.items() if (match := pattern.search(output)) is not None
    ]
    if len(matches) != 1:
        if not matches:
            raise ActionParseError(f"no supported action found in model output: {output!r}")
        raise ActionParseError(f"ambiguous model output contains multiple actions: {output!r}")

    action, action_match = matches[0]
    suffix = output[action_match.end() :]
    if action is ActionType.STOP:
        if not _PUNCTUATION_ONLY.fullmatch(suffix):
            raise ActionParseError(f"malformed stop action in model output: {output!r}")
        return ParsedAction(type=action, amount=0, vx=0.0, vy=0.0, vyaw=0.0, duration_sec=0.0)

    amount_match = _AMOUNT_SUFFIX_PATTERNS[action].match(suffix)
    if amount_match is None:
        if not _PUNCTUATION_ONLY.fullmatch(suffix):
            raise ActionParseError(f"malformed action amount in model output: {output!r}")
        raw_amount = None
    else:
        if not _PUNCTUATION_ONLY.fullmatch(suffix[amount_match.end() :]):
            raise ActionParseError(f"malformed action amount in model output: {output!r}")
        raw_amount = int(amount_match.group(1))

    if action is ActionType.FORWARD:
        amount = _nearest(raw_amount or FORWARD_AMOUNTS_CM[0], FORWARD_AMOUNTS_CM)
        velocity = config.motion.forward_velocity_mps
        return ParsedAction(
            type=action,
            amount=amount,
            vx=velocity,
            vy=0.0,
            vyaw=0.0,
            duration_sec=(amount / 100.0) / velocity,
        )

    amount = _nearest(raw_amount or TURN_AMOUNTS_DEGREES[0], TURN_AMOUNTS_DEGREES)
    yaw_velocity = config.motion.turn_velocity_rps
    if action is ActionType.RIGHT:
        yaw_velocity = -yaw_velocity
    return ParsedAction(
        type=action,
        amount=amount,
        vx=0.0,
        vy=0.0,
        vyaw=yaw_velocity,
        duration_sec=math.radians(amount) / config.motion.turn_velocity_rps,
    )


class Controller:
    """Own all mutable node state behind one lock."""

    def __init__(
        self,
        config: NodeConfig,
        *,
        num_video_frames: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if isinstance(num_video_frames, bool) or not isinstance(num_video_frames, int) or num_video_frames <= 0:
            raise ValueError("model.config.num_video_frames must be a positive integer")

        self._config = config
        self._clock = clock
        self._lock = RLock()
        self._lifecycle = Lifecycle.INITIALIZING
        self._instruction = ""
        self._frames: deque[Image.Image] = deque(maxlen=num_video_frames)
        self._last_frame_at: float | None = None
        self._last_heartbeat_at: float | None = None
        self._go2_state: Go2NodeState | None = None
        self._last_go2_state_at: float | None = None
        self._inference_active = False
        self._last_inference_duration: float | None = None
        self._last_inference_output: str | None = None
        self._current_action: ParsedAction | None = None
        self._action_deadline: float | None = None
        self._stand_deadline: float | None = None
        self._down_deadline: float | None = None
        self._last_error: str | None = None
        self._stop_error: str | None = None
        self._zero_pending = False
        self._posture_pending: Literal["stand", "down"] | None = None
        self._startup_seat_pending = True
        self._shutdown_requested = False
        self._shutdown_complete = False

    @property
    def frame_capacity(self) -> int:
        assert self._frames.maxlen is not None
        return self._frames.maxlen

    @property
    def shutdown_complete(self) -> bool:
        with self._lock:
            return self._shutdown_complete

    @property
    def shutdown_error(self) -> str | None:
        with self._lock:
            return self._last_error if self._shutdown_complete and self._lifecycle is Lifecycle.ERROR else None

    def initialize(self) -> None:
        with self._lock:
            if self._lifecycle is not Lifecycle.INITIALIZING:
                raise RuntimeError("controller has already been initialized")
            self._lifecycle = Lifecycle.IDLE

    def camera_sample_due(self, now: float | None = None) -> bool:
        now = self._clock() if now is None else now
        with self._lock:
            if self._last_frame_at is None:
                return True
            return now - self._last_frame_at >= 1.0 / self._config.camera.sample_frequency_hz

    def append_camera_frame(self, frame: Image.Image, *, sampled_at: float | None = None) -> bool:
        sampled_at = self._clock() if sampled_at is None else sampled_at
        with self._lock:
            if self._last_frame_at is not None:
                period = 1.0 / self._config.camera.sample_frequency_hz
                if sampled_at - self._last_frame_at < period:
                    return False
            self._frames.append(frame)
            self._last_frame_at = sampled_at
            return True

    def update_go2_state(self, state: Go2NodeState, *, received_at: float | None = None) -> None:
        received_at = self._clock() if received_at is None else received_at
        with self._lock:
            self._go2_state = state
            self._last_go2_state_at = received_at

    def handle_command(
        self,
        command: HeartbeatCommand | InstructionCommand | StartCommand | StopCommand,
        *,
        received_at: float | None = None,
    ) -> bool:
        received_at = self._clock() if received_at is None else received_at
        with self._lock:
            if isinstance(command, HeartbeatCommand):
                self._last_heartbeat_at = received_at
                return True
            if isinstance(command, StopCommand):
                return self._request_operator_stop(received_at)
            if not self._cli_connected(received_at):
                return False
            if isinstance(command, InstructionCommand):
                if self._lifecycle not in (Lifecycle.IDLE, Lifecycle.ERROR) or self._inference_active:
                    return False
                self._instruction = command.instruction
                return True
            assert isinstance(command, StartCommand)
            return self._request_start(received_at)

    def begin_inference(self, *, now: float | None = None) -> InferenceRequest | None:
        now = self._clock() if now is None else now
        with self._lock:
            if self._lifecycle is not Lifecycle.RUNNING or self._inference_active:
                return None
            if self._camera_stale(now) or not self._frames:
                return None
            self._inference_active = True
            return InferenceRequest(self._instruction, tuple(self._frames))

    def complete_inference(self, output: str, *, duration_sec: float, completed_at: float | None = None) -> bool:
        completed_at = self._clock() if completed_at is None else completed_at
        with self._lock:
            if not self._inference_active:
                raise RuntimeError("no inference is active")
            self._inference_active = False
            if self._lifecycle is not Lifecycle.RUNNING:
                return False
            self._last_inference_duration = max(0.0, duration_sec)
            self._last_inference_output = output
            try:
                action = parse_navigation_action(output, self._config)
            except ActionParseError as error:
                self._begin_stop(
                    completed_at,
                    error=str(error),
                    request_down=self._go2_available(completed_at),
                )
                return False
            if action.type is ActionType.STOP:
                self._begin_stop(
                    completed_at,
                    error=None,
                    request_down=self._go2_available(completed_at),
                )
                return False
            self._current_action = action
            self._action_deadline = completed_at + action.duration_sec
            return True

    def fail_inference(self, error: Exception, *, failed_at: float | None = None) -> None:
        failed_at = self._clock() if failed_at is None else failed_at
        with self._lock:
            was_active = self._inference_active
            self._inference_active = False
            if was_active and self._lifecycle in (Lifecycle.RUNNING, Lifecycle.STOPPING):
                self._begin_stop(
                    failed_at,
                    error=f"inference failed: {error}",
                    request_down=self._go2_available(failed_at),
                )

    def request_shutdown(self, *, now: float | None = None) -> None:
        now = self._clock() if now is None else now
        with self._lock:
            if self._shutdown_requested:
                return
            self._shutdown_requested = True
            request_down = self._go2_available(now)
            error = None if request_down else "cannot confirm Go2 state during shutdown"
            self._begin_stop(now, error=error, request_down=request_down, allow_idle=True)

    def advance(self, *, now: float | None = None) -> None:
        now = self._clock() if now is None else now
        with self._lock:
            if self._lifecycle in (Lifecycle.STANDING, Lifecycle.RUNNING):
                if not self._cli_connected(now):
                    self._begin_stop(now, error=None, request_down=self._go2_available(now))
                elif self._camera_stale(now):
                    self._begin_stop(now, error="camera input is stale", request_down=self._go2_available(now))
                elif not self._go2_available(now):
                    self._begin_stop(now, error="Go2 state is stale or disconnected", request_down=False)

            if self._lifecycle is Lifecycle.STANDING:
                if self._go2_ready_to_move():
                    self._lifecycle = Lifecycle.RUNNING
                    self._stand_deadline = None
                elif self._stand_deadline is not None and now >= self._stand_deadline:
                    self._begin_stop(now, error="Go2 did not become ready_stand before timeout", request_down=True)
            elif self._lifecycle is Lifecycle.RUNNING:
                self._expire_action(now)
            elif self._lifecycle is Lifecycle.STOPPING:
                self._advance_stop(now)
            elif self._lifecycle is Lifecycle.IDLE:
                self._advance_startup_seating(now)

    def take_effects(self) -> ControlEffects:
        with self._lock:
            effects = ControlEffects(zero_velocity=self._zero_pending, posture=self._posture_pending)
            self._zero_pending = False
            self._posture_pending = None
            return effects

    def velocity_for_publish(self, *, now: float | None = None) -> VelocityCommand | None:
        now = self._clock() if now is None else now
        with self._lock:
            if self._lifecycle is not Lifecycle.RUNNING:
                return None
            self._expire_action(now)
            if self._current_action is None:
                return ZERO_VELOCITY
            return VelocityCommand(
                vx=self._current_action.vx,
                vy=self._current_action.vy,
                vyaw=self._current_action.vyaw,
            )

    def snapshot(self, *, now: float | None = None) -> NodeState:
        now = self._clock() if now is None else now
        with self._lock:
            self._expire_action(now)
            ready_reason = self._readiness_reason(now)
            action_state = None
            if self._lifecycle is Lifecycle.RUNNING and self._current_action is not None:
                assert self._action_deadline is not None
                action_state = ActionState(
                    type=self._current_action.type,
                    vx=self._current_action.vx,
                    vy=self._current_action.vy,
                    vyaw=self._current_action.vyaw,
                    remaining_sec=max(0.0, self._action_deadline - now),
                )
            return NodeState(
                lifecycle=self._lifecycle,
                cli_connected=self._cli_connected(now),
                start_ready=ready_reason is None,
                status_message=self._status_message(ready_reason),
                instruction=self._instruction,
                inference=InferenceState(
                    active=self._inference_active,
                    last_duration_sec=self._last_inference_duration,
                    last_output=self._last_inference_output,
                ),
                action=action_state,
                last_error=self._last_error,
            )

    def _request_operator_stop(self, now: float) -> bool:
        if self._lifecycle in (Lifecycle.STANDING, Lifecycle.RUNNING):
            self._begin_stop(now, error=None, request_down=self._go2_available(now))
            return True
        if self._lifecycle is Lifecycle.STOPPING:
            return True
        return self._lifecycle in (Lifecycle.IDLE, Lifecycle.ERROR)

    def _request_start(self, now: float) -> bool:
        if self._readiness_reason(now) is not None:
            return False
        self._lifecycle = Lifecycle.STANDING
        self._last_error = None
        self._stop_error = None
        self._current_action = None
        self._action_deadline = None
        self._stand_deadline = now + self._config.go2.stand_timeout_seconds
        self._posture_pending = "stand"
        return True

    def _begin_stop(
        self,
        now: float,
        *,
        error: str | None,
        request_down: bool,
        allow_idle: bool = False,
    ) -> None:
        if self._lifecycle is Lifecycle.STOPPING:
            if error is not None and self._stop_error is None:
                self._stop_error = error
            return
        if self._lifecycle in (Lifecycle.IDLE, Lifecycle.ERROR) and not allow_idle:
            return
        self._lifecycle = Lifecycle.STOPPING
        self._stop_error = error
        self._current_action = None
        self._action_deadline = None
        self._stand_deadline = None
        self._down_deadline = now + self._config.go2.down_timeout_seconds
        self._zero_pending = True
        self._posture_pending = "down" if request_down else None

    def _advance_stop(self, now: float) -> None:
        if not self._go2_available(now):
            self._posture_pending = None
            self._finish_stop(error=self._stop_error or "Go2 state became stale before down was confirmed")
            return
        if self._go2_state is not None and self._go2_state.robot_state.state == "down":
            self._startup_seat_pending = False
            if not self._inference_active:
                self._finish_stop(error=self._stop_error)
            return
        if self._down_deadline is not None and now >= self._down_deadline:
            self._finish_stop(error=self._stop_error or "Go2 did not confirm down before timeout")

    def _finish_stop(self, *, error: str | None) -> None:
        self._down_deadline = None
        self._last_error = error
        self._lifecycle = Lifecycle.ERROR if error is not None else Lifecycle.IDLE
        if self._shutdown_requested:
            self._shutdown_complete = True

    def _advance_startup_seating(self, now: float) -> None:
        if not self._startup_seat_pending or not self._go2_available(now):
            return
        assert self._go2_state is not None
        if self._go2_state.robot_state.state == "down":
            self._startup_seat_pending = False
            return
        self._begin_stop(now, error=None, request_down=True, allow_idle=True)

    def _expire_action(self, now: float) -> None:
        if self._action_deadline is not None and now >= self._action_deadline:
            self._current_action = None
            self._action_deadline = None

    def _cli_connected(self, now: float) -> bool:
        return (
            self._last_heartbeat_at is not None
            and now - self._last_heartbeat_at <= self._config.control.cli_heartbeat_timeout_seconds
        )

    def _camera_stale(self, now: float) -> bool:
        return (
            self._last_frame_at is None
            or now - self._last_frame_at > self._config.camera.frame_freshness_seconds
        )

    def _go2_fresh(self, now: float) -> bool:
        return (
            self._last_go2_state_at is not None
            and now - self._last_go2_state_at <= self._config.go2.node_state_timeout_seconds
        )

    def _go2_available(self, now: float) -> bool:
        return self._go2_fresh(now) and self._go2_state is not None and self._go2_state.robot_connected

    def _go2_ready_to_move(self) -> bool:
        return (
            self._go2_state is not None
            and self._go2_state.robot_connected
            and self._go2_state.accepting_commands
            and self._go2_state.robot_state.state == "ready_stand"
        )

    def _readiness_reason(self, now: float) -> str | None:
        if self._lifecycle not in (Lifecycle.IDLE, Lifecycle.ERROR):
            return f"node is {self._lifecycle.value}"
        if self._inference_active:
            return "waiting for the previous inference to finish"
        if not self._cli_connected(now):
            return "waiting for CLI heartbeat"
        if not self._instruction:
            return "instruction is empty"
        if not self._frames:
            return "waiting for a valid camera frame"
        if self._camera_stale(now):
            return "camera input is stale"
        if not self._go2_fresh(now):
            return "waiting for fresh Go2 state"
        assert self._go2_state is not None
        if not self._go2_state.robot_connected:
            return "Go2 node reports disconnected"
        if not self._go2_state.accepting_commands:
            return "Go2 node is not accepting commands"
        if self._go2_state.robot_state.state != "down":
            return "waiting for Go2 to be down"
        return None

    def _status_message(self, readiness_reason: str | None) -> str | None:
        if self._lifecycle is Lifecycle.INITIALIZING:
            return "initializing"
        if self._lifecycle is Lifecycle.STANDING:
            return "waiting for Go2 ready_stand"
        if self._lifecycle is Lifecycle.RUNNING:
            return "running"
        if self._lifecycle is Lifecycle.STOPPING:
            return "stopping and seating Go2"
        if self._lifecycle is Lifecycle.ERROR:
            return self._last_error or readiness_reason or "error"
        return readiness_reason or "ready"
