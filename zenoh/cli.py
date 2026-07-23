"""Interactive Zenoh operator client for the NaVILA inference node."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from pathlib import Path
from threading import Condition, Event, Thread
from typing import Any

from config import (
    JSON_ENCODING,
    CliConfig,
    HeartbeatCommand,
    InstructionCommand,
    Lifecycle,
    NodeState,
    StartCommand,
    StopCommand,
    decode_node_state,
    load_cli_config,
)
from pydantic import ValidationError

import zenoh

DEFAULT_CLI_CONFIG_PATH = Path("zenoh/cli-config.json5")
DEFAULT_ZENOH_CONFIG_PATH = Path("zenoh/cli-zenoh-config.json5")
LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="navila-zenoh-cli",
        description="Set an instruction and control a NaVILA Zenoh node.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cli-config",
        type=Path,
        default=DEFAULT_CLI_CONFIG_PATH,
        metavar="FILE",
        help="CLI JSON5 configuration file.",
    )
    parser.add_argument(
        "--zenoh-config",
        type=Path,
        default=DEFAULT_ZENOH_CONFIG_PATH,
        metavar="FILE",
        help="Zenoh JSON5 configuration file.",
    )
    return parser


def _json_payload(model: Any) -> str:
    return model.json(separators=(",", ":"))


def _has_json_encoding(sample: Any) -> bool:
    encoding = str(getattr(sample, "encoding", ""))
    return encoding == JSON_ENCODING or encoding.startswith(f"{JSON_ENCODING};")


class StateCache:
    """Validate state callbacks and provide revision-based command acknowledgements."""

    def __init__(
        self,
        timeout_seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._timeout_seconds = timeout_seconds
        self._clock = clock
        self._condition = Condition()
        self._revision = 0
        self._state: NodeState | None = None
        self._received_at: float | None = None

    def update(self, sample: Any) -> None:
        try:
            if not _has_json_encoding(sample):
                raise ValueError(f"expected {JSON_ENCODING}, got {sample.encoding}")
            state = decode_node_state(sample.payload.to_bytes())
        except (UnicodeDecodeError, ValidationError, ValueError) as error:
            LOGGER.warning("Rejected invalid NaVILA state: %s", error)
            return
        self.store(state)

    def store(self, state: NodeState, *, received_at: float | None = None) -> None:
        """Store validated state; exposed separately to keep tests independent of Zenoh samples."""

        received_at = self._clock() if received_at is None else received_at
        with self._condition:
            self._revision += 1
            self._state = state
            self._received_at = received_at
            self._condition.notify_all()

    def snapshot(self, *, now: float | None = None) -> tuple[int, NodeState | None, bool]:
        now = self._clock() if now is None else now
        with self._condition:
            fresh = self._received_at is not None and now - self._received_at <= self._timeout_seconds
            return self._revision, self._state, fresh

    def wait_after(
        self,
        revision: int,
        predicate: Callable[[NodeState], bool],
        *,
        timeout_seconds: float,
    ) -> NodeState | None:
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while True:
                if self._revision > revision and self._state is not None and predicate(self._state):
                    return self._state
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)


class CliRuntime:
    """Own the CLI session, heartbeat, prompts, and state-based acknowledgements."""

    def __init__(
        self,
        config: CliConfig,
        *,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
    ) -> None:
        self._config = config
        self._input = input_fn
        self._output = output_fn
        self._states = StateCache(config.node_state_timeout_seconds)
        self._heartbeat_publisher: Any = None
        self._command_publisher: Any = None
        self._stop_heartbeat = Event()
        self._heartbeat_error: Exception | None = None

    def run(self, zenoh_config: zenoh.Config) -> None:
        zenoh.init_log_from_env_or("error")
        command_key = f"{self._config.zenoh_key_prefix}/command"
        state_key = f"{self._config.zenoh_key_prefix}/state"

        with zenoh.open(zenoh_config) as session, ExitStack() as resources:
            self._heartbeat_publisher = resources.enter_context(
                session.declare_publisher(
                    command_key,
                    encoding=JSON_ENCODING,
                    congestion_control=zenoh.CongestionControl.DROP,
                    reliability=zenoh.Reliability.BEST_EFFORT,
                )
            )
            self._command_publisher = resources.enter_context(
                session.declare_publisher(
                    command_key,
                    encoding=JSON_ENCODING,
                    congestion_control=zenoh.CongestionControl.BLOCK,
                    reliability=zenoh.Reliability.RELIABLE,
                )
            )
            resources.enter_context(session.declare_subscriber(state_key, self._states.update))

            heartbeat = Thread(target=self._run_heartbeat, name="navila-cli-heartbeat", daemon=True)
            heartbeat.start()
            try:
                self._interactive_loop()
            except (EOFError, KeyboardInterrupt):
                self._output("")
                self._stop_before_exit()
            finally:
                self._stop_heartbeat.set()
                heartbeat.join(self._config.heartbeat_interval_seconds + 1.0)
                if heartbeat.is_alive():
                    raise RuntimeError("CLI heartbeat worker did not stop")

        if self._heartbeat_error is not None:
            raise RuntimeError(f"heartbeat worker failed: {self._heartbeat_error}") from self._heartbeat_error

    def _interactive_loop(self) -> None:
        self._output("Commands: ins <instruction>, start, status. Ctrl+C exits.")
        while True:
            _revision, state, fresh = self._states.snapshot()
            if fresh and state is not None and state.lifecycle is Lifecycle.RUNNING:
                self._input("Running - press Enter to stop")
                self._request_stop(report_timeout=True)
                continue

            line = self._input("navila> ").strip()
            if not line:
                continue
            if line == "status":
                self._print_status()
            elif line == "start":
                self._request_start()
            elif line == "ins" or line.startswith("ins "):
                instruction = line[3:].strip()
                self._set_instruction(instruction)
            else:
                self._output("Unknown command. Use: ins <instruction>, start, status")

    def _run_heartbeat(self) -> None:
        heartbeat = _json_payload(HeartbeatCommand())
        try:
            while True:
                self._heartbeat_publisher.put(heartbeat)
                if self._stop_heartbeat.wait(self._config.heartbeat_interval_seconds):
                    return
        except Exception as error:
            self._heartbeat_error = error
            LOGGER.error("CLI heartbeat failed: %s", error)

    def _set_instruction(self, instruction: str) -> None:
        revision, state = self._require_fresh_state()
        if state is None:
            return
        if state.lifecycle not in (Lifecycle.IDLE, Lifecycle.ERROR) or state.inference.active:
            self._output("Stop navigation before changing the instruction.")
            return
        if not state.cli_connected:
            self._output("Waiting for the CLI heartbeat to reach the NaVILA node.")
            return

        try:
            command = InstructionCommand(instruction=instruction)
        except ValidationError as error:
            self._output(f"Invalid instruction: {error.errors()[0]['msg']}")
            return
        self._command_publisher.put(_json_payload(command))
        acknowledged = self._states.wait_after(
            revision,
            lambda update: update.instruction == command.instruction,
            timeout_seconds=self._config.command_timeout_seconds,
        )
        if acknowledged is None:
            self._output("Instruction was not acknowledged before the timeout.")
        elif command.instruction:
            self._output(f"Instruction: {command.instruction}")
        else:
            self._output("Instruction cleared.")

    def _request_start(self) -> None:
        revision, state = self._require_fresh_state()
        if state is None:
            return
        if not state.start_ready:
            self._output(f"Cannot start: {state.status_message or 'node is not ready'}")
            return

        self._command_publisher.put(_json_payload(StartCommand()))
        self._output("Start requested; waiting for Go2 ready_stand.")
        acknowledged = self._states.wait_after(
            revision,
            lambda update: update.lifecycle in (Lifecycle.RUNNING, Lifecycle.ERROR),
            timeout_seconds=self._config.command_timeout_seconds,
        )
        if acknowledged is None:
            self._output("Start was not acknowledged before the timeout.")
        elif acknowledged.lifecycle is Lifecycle.ERROR:
            self._output(f"Start failed: {acknowledged.last_error or acknowledged.status_message or 'unknown error'}")
        else:
            self._output("Navigation started.")

    def _request_stop(self, *, report_timeout: bool) -> bool:
        revision, _state, _fresh = self._states.snapshot()
        self._command_publisher.put(_json_payload(StopCommand()))
        stopped = self._states.wait_after(
            revision,
            lambda update: update.lifecycle in (Lifecycle.IDLE, Lifecycle.ERROR) and not update.inference.active,
            timeout_seconds=self._config.command_timeout_seconds,
        )
        if stopped is None:
            if report_timeout:
                self._output("Stop was not confirmed before the timeout.")
            return False
        if stopped.lifecycle is Lifecycle.ERROR:
            self._output(
                f"Stopped with error: {stopped.last_error or 'Go2 resting state was not confirmed'}"
            )
        else:
            self._output("Stopped; Go2 is resting.")
        return True

    def _stop_before_exit(self) -> None:
        _revision, state, fresh = self._states.snapshot()
        safely_idle = fresh and state is not None and state.lifecycle is Lifecycle.IDLE and not state.inference.active
        if safely_idle:
            return
        self._output("Requesting safe stop before exit...")
        self._request_stop(report_timeout=True)

    def _require_fresh_state(self) -> tuple[int, NodeState | None]:
        revision, state, fresh = self._states.snapshot()
        if not fresh or state is None:
            self._output(f"NaVILA node is disconnected (no state for {self._config.node_state_timeout_seconds:g}s).")
            return revision, None
        return revision, state

    def _print_status(self) -> None:
        _revision, state = self._require_fresh_state()
        if state is None:
            return
        instruction = state.instruction or "<empty>"
        self._output(
            f"lifecycle={state.lifecycle.value} connected={state.cli_connected} "
            f"ready={state.start_ready} instruction={instruction!r}"
        )
        if state.status_message:
            self._output(f"status: {state.status_message}")
        if state.last_error:
            self._output(f"error: {state.last_error}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    try:
        cli_config = load_cli_config(args.cli_config)
        zenoh_config = zenoh.Config.from_file(args.zenoh_config)
        CliRuntime(cli_config).run(zenoh_config)
    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")
    except (OSError, RuntimeError, ValueError, ValidationError, zenoh.ZError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
