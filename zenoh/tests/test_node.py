from __future__ import annotations

from contextlib import nullcontext
from io import BytesIO
from threading import Event
from unittest.mock import MagicMock, patch

import pytest
from config import (
    Go2NodeState,
    Go2RobotState,
    HeartbeatCommand,
    InstructionCommand,
    NodeConfig,
    StartCommand,
)
from node import (
    NaVILAEngine,
    NodeRuntime,
    build_navigation_question,
    next_future_deadline,
    prepare_inference_frames,
)
from PIL import Image

import zenoh


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakePayload:
    def __init__(self, value: bytes) -> None:
        self._value = value

    def to_bytes(self) -> bytes:
        return self._value


class FakeSample:
    def __init__(self, payload: bytes, encoding: str) -> None:
        self.payload = FakePayload(payload)
        self.encoding = encoding


class FakeEngine:
    num_video_frames = 4

    def infer(self, _request: object) -> str:
        return "The next action is stop"


class TimedStop:
    def __init__(self, clock: FakeClock, stop_at: float) -> None:
        self._clock = clock
        self._stop_at = stop_at

    def is_set(self) -> bool:
        return self._clock.now >= self._stop_at

    def wait(self, timeout: float) -> bool:
        self._clock.now += timeout
        return self.is_set()


class AdvancingWake:
    def __init__(self, clock: FakeClock) -> None:
        self._clock = clock

    def wait(self, timeout: float) -> bool:
        self._clock.now += timeout
        return False

    def clear(self) -> None:
        pass


def jpeg_bytes(color: str = "white") -> bytes:
    output = BytesIO()
    Image.new("RGB", (8, 6), color).save(output, format="JPEG")
    return output.getvalue()


def prepare_running_runtime(runtime: NodeRuntime) -> None:
    runtime.controller.initialize()
    runtime.controller.append_camera_frame(Image.new("RGB", (8, 8), "white"))
    runtime.controller.update_go2_state(
        Go2NodeState(
            robot_connected=True,
            robot_state=Go2RobotState(state="down"),
            accepting_commands=True,
        )
    )
    runtime.controller.advance()
    runtime.controller.handle_command(HeartbeatCommand())
    runtime.controller.handle_command(InstructionCommand(instruction="go"))
    assert runtime.controller.handle_command(StartCommand())
    runtime.controller.take_effects()
    runtime.controller.update_go2_state(
        Go2NodeState(
            robot_connected=True,
            robot_state=Go2RobotState(state="ready_stand"),
            accepting_commands=True,
        )
    )
    runtime.controller.advance()


def test_absolute_schedule_skips_missed_slots_without_catch_up() -> None:
    assert next_future_deadline(10.0, 1.0, 9.0) == 10.0
    assert next_future_deadline(10.0, 1.0, 10.0) == 11.0
    assert next_future_deadline(10.0, 1.0, 12.4) == 13.0


def test_frame_preparation_left_pads_black_and_preserves_current_observation() -> None:
    old = Image.new("RGB", (16, 12), "red")
    current = Image.new("RGB", (16, 12), "blue")

    frames = prepare_inference_frames((old, current), 4)

    assert len(frames) == 4
    assert frames[0].size == (512, 512)
    assert frames[0].getpixel((0, 0)) == (0, 0, 0)
    assert frames[1].getpixel((0, 0)) == (0, 0, 0)
    assert frames[2] is old
    assert frames[3] is current


def test_frame_preparation_keeps_only_latest_model_sized_history() -> None:
    frames = tuple(Image.new("RGB", (2, 2), (index, 0, 0)) for index in range(5))

    prepared = prepare_inference_frames(frames, 3)

    assert [frame.getpixel((0, 0))[0] for frame in prepared] == [2, 3, 4]
    with pytest.raises(ValueError, match="real camera frame"):
        prepare_inference_frames((), 4)


def test_navigation_question_matches_frame_count_without_action_history() -> None:
    question = build_navigation_question("Go to the elevator", 4)

    assert question.count("<image>") == 4
    assert 'Your assigned task is: "Go to the elevator"' in question
    assert "previous action" not in question.lower()


def test_camera_callback_decodes_only_due_valid_jpegs(node_config: NodeConfig) -> None:
    clock = FakeClock()
    runtime = NodeRuntime(node_config, FakeEngine(), clock=clock)

    runtime._on_camera(FakeSample(jpeg_bytes("red"), "image/jpeg"))
    assert not runtime.controller.camera_sample_due()
    clock.now = 0.5
    runtime._on_camera(FakeSample(jpeg_bytes("blue"), "image/jpeg"))
    assert not runtime.controller.camera_sample_due()


def test_control_worker_publishes_state_and_velocity_at_20_hz(node_config: NodeConfig) -> None:
    clock = FakeClock()
    runtime = NodeRuntime(node_config, FakeEngine(), clock=clock)
    prepare_running_runtime(runtime)
    runtime._velocity_publisher = MagicMock()
    runtime._posture_publisher = MagicMock()
    runtime._state_publisher = MagicMock()
    runtime._stop_workers = TimedStop(clock, stop_at=0.16)
    runtime._wake_control = AdvancingWake(clock)

    runtime._run_control_worker()

    assert runtime._velocity_publisher.put.call_count == 4
    assert runtime._state_publisher.put.call_count == 4
    assert runtime._posture_publisher.put.call_count == 0


def test_inference_worker_is_serial_and_skips_overrun_slots(node_config: NodeConfig) -> None:
    clock = FakeClock()

    class OverrunningEngine(FakeEngine):
        def __init__(self) -> None:
            self.started_at: list[float] = []

        def infer(self, _request: object) -> str:
            self.started_at.append(clock.now)
            clock.now += 1.4
            return "The next action is move forward 25 cm"

    engine = OverrunningEngine()
    runtime = NodeRuntime(node_config, engine, clock=clock)
    prepare_running_runtime(runtime)
    runtime._stop_workers = TimedStop(clock, stop_at=4.0)

    runtime._run_inference_worker()

    assert engine.started_at == pytest.approx([0.0, 2.0])
    clock.now = 1.0
    runtime._on_camera(FakeSample(b"not-a-jpeg", "image/jpeg"))
    assert runtime.controller.camera_sample_due()
    runtime._on_camera(FakeSample(jpeg_bytes("blue"), "application/octet-stream"))
    assert runtime.controller.camera_sample_due()
    runtime._on_camera(FakeSample(jpeg_bytes("blue"), "image/jpeg"))
    assert not runtime.controller.camera_sample_due()


def test_runtime_declares_separate_qos_and_uses_safe_shutdown(node_config: NodeConfig) -> None:
    runtime = NodeRuntime(node_config, FakeEngine())
    runtime.controller.update_go2_state(
        Go2NodeState(
            robot_connected=True,
            robot_state=Go2RobotState(state="down"),
            accepting_commands=True,
        )
    )

    velocity_publisher = MagicMock()
    posture_publisher = MagicMock()
    state_publisher = MagicMock()
    publisher_contexts = [
        nullcontext(velocity_publisher),
        nullcontext(posture_publisher),
        nullcontext(state_publisher),
    ]
    session = MagicMock()
    session.declare_publisher.side_effect = publisher_contexts
    session.declare_subscriber.side_effect = [nullcontext(), nullcontext(), nullcontext()]
    open_context = MagicMock()
    open_context.__enter__.return_value = session
    open_context.__exit__.return_value = False
    shutdown = Event()
    shutdown.set()

    with patch("node.zenoh.open", return_value=open_context):
        runtime.run(MagicMock(), shutdown)

    velocity_call, posture_call, state_call = session.declare_publisher.call_args_list
    assert velocity_call.args[0] == "unitree/go2/command"
    assert velocity_call.kwargs["congestion_control"] is zenoh.CongestionControl.DROP
    assert velocity_call.kwargs["reliability"] is zenoh.Reliability.BEST_EFFORT
    assert posture_call.args[0] == "unitree/go2/command"
    assert posture_call.kwargs["congestion_control"] is zenoh.CongestionControl.BLOCK
    assert posture_call.kwargs["reliability"] is zenoh.Reliability.RELIABLE
    assert state_call.args[0] == "navila/state"
    assert velocity_publisher.put.called
    assert posture_publisher.put.called
    assert state_publisher.put.called


def test_model_frame_count_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        NaVILAEngine(MagicMock(), MagicMock(), MagicMock(), num_video_frames=0)
