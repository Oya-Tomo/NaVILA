from __future__ import annotations

import json

from cli import CliRuntime, StateCache
from config import (
    CliConfig,
    InferenceState,
    Lifecycle,
    NodeState,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class RecordingPublisher:
    def __init__(self, on_put=None) -> None:
        self.payloads: list[dict[str, object]] = []
        self._on_put = on_put

    def put(self, payload: str) -> None:
        document = json.loads(payload)
        self.payloads.append(document)
        if self._on_put is not None:
            self._on_put(document)


def cli_config() -> CliConfig:
    return CliConfig(
        node_key="navila",
        heartbeat_interval_sec=0.2,
        node_state_stale_timeout_sec=1.0,
        command_timeout_sec=15.0,
    )


def state(
    lifecycle: Lifecycle = Lifecycle.IDLE,
    *,
    instruction: str = "",
    ready: bool = False,
    inference_active: bool = False,
    error: str | None = None,
    cli_connected: bool = True,
) -> NodeState:
    return NodeState(
        lifecycle=lifecycle,
        cli_connected=cli_connected,
        start_ready=ready,
        status_message="ready" if ready else lifecycle.value,
        instruction=instruction,
        inference=InferenceState(
            active=inference_active,
            last_duration_sec=None,
            last_output=None,
        ),
        action=None,
        last_error=error,
    )


def test_state_cache_tracks_freshness_and_revision() -> None:
    clock = FakeClock()
    cache = StateCache(1.0, clock=clock)
    assert cache.snapshot() == (0, None, False)

    cache.store(state(), received_at=0.0)
    revision, current, fresh = cache.snapshot()
    assert revision == 1
    assert current is not None
    assert fresh

    clock.now = 1.01
    assert cache.snapshot()[2] is False


def test_instruction_uses_post_publish_state_as_acknowledgement() -> None:
    outputs: list[str] = []
    runtime = CliRuntime(cli_config(), output_fn=outputs.append)
    runtime._states.store(state(ready=False))

    def acknowledge(document: dict[str, object]) -> None:
        runtime._states.store(state(instruction=str(document["instruction"])))

    publisher = RecordingPublisher(acknowledge)
    runtime._command_publisher = publisher
    runtime._set_instruction("  Go to the elevator  ")

    assert publisher.payloads == [{"type": "instruction", "instruction": "Go to the elevator"}]
    assert outputs[-1] == "Instruction: Go to the elevator"

    runtime._set_instruction("")
    assert publisher.payloads[-1] == {"type": "instruction", "instruction": ""}
    assert outputs[-1] == "Instruction cleared."


def test_instruction_change_is_rejected_while_running() -> None:
    outputs: list[str] = []
    runtime = CliRuntime(cli_config(), output_fn=outputs.append)
    runtime._states.store(state(Lifecycle.RUNNING, instruction="old"))
    publisher = RecordingPublisher()
    runtime._command_publisher = publisher

    runtime._set_instruction("new")

    assert publisher.payloads == []
    assert "Stop navigation" in outputs[-1]


def test_instruction_waits_for_cli_heartbeat_acknowledgement() -> None:
    outputs: list[str] = []
    runtime = CliRuntime(cli_config(), output_fn=outputs.append)
    runtime._states.store(state(cli_connected=False))
    publisher = RecordingPublisher()
    runtime._command_publisher = publisher

    runtime._set_instruction("new")

    assert publisher.payloads == []
    assert "heartbeat" in outputs[-1]


def test_start_and_stop_wait_for_node_state_acknowledgements() -> None:
    outputs: list[str] = []
    runtime = CliRuntime(cli_config(), output_fn=outputs.append)
    runtime._states.store(state(instruction="go", ready=True))

    def acknowledge(document: dict[str, object]) -> None:
        if document["type"] == "start":
            runtime._states.store(state(Lifecycle.RUNNING, instruction="go"))
        elif document["type"] == "stop":
            runtime._states.store(state(Lifecycle.IDLE, instruction="go", ready=True))

    publisher = RecordingPublisher(acknowledge)
    runtime._command_publisher = publisher

    runtime._request_start()
    assert publisher.payloads[-1] == {"type": "start"}
    assert outputs[-1] == "Navigation started."

    assert runtime._request_stop(report_timeout=True)
    assert publisher.payloads[-1] == {"type": "stop"}
    assert outputs[-1] == "Stopped; Go2 is down."


def test_start_refuses_disconnected_or_not_ready_node() -> None:
    outputs: list[str] = []
    runtime = CliRuntime(cli_config(), output_fn=outputs.append)
    runtime._command_publisher = RecordingPublisher()

    runtime._request_start()
    assert "disconnected" in outputs[-1]

    runtime._states.store(state(ready=False))
    runtime._request_start()
    assert outputs[-1] == "Cannot start: idle"
    assert runtime._command_publisher.payloads == []


def test_running_enter_sends_stop_and_returns_to_prompt() -> None:
    prompts: list[str] = []
    inputs = iter(["", KeyboardInterrupt()])

    def input_fn(prompt: str) -> str:
        prompts.append(prompt)
        value = next(inputs)
        if isinstance(value, BaseException):
            raise value
        return value

    runtime = CliRuntime(cli_config(), input_fn=input_fn, output_fn=lambda _message: None)
    runtime._states.store(state(Lifecycle.RUNNING, instruction="go"))

    def acknowledge(document: dict[str, object]) -> None:
        if document["type"] == "stop":
            runtime._states.store(state(Lifecycle.IDLE, instruction="go", ready=True))

    publisher = RecordingPublisher(acknowledge)
    runtime._command_publisher = publisher

    try:
        runtime._interactive_loop()
    except KeyboardInterrupt:
        pass

    assert prompts == ["Running - press Enter to stop", "navila> "]
    assert publisher.payloads == [{"type": "stop"}]


def test_ctrl_c_exit_sends_stop_unless_fresh_idle() -> None:
    outputs: list[str] = []
    runtime = CliRuntime(cli_config(), output_fn=outputs.append)

    def acknowledge(document: dict[str, object]) -> None:
        if document["type"] == "stop":
            runtime._states.store(state(Lifecycle.IDLE))

    publisher = RecordingPublisher(acknowledge)
    runtime._command_publisher = publisher
    runtime._states.store(state(Lifecycle.STANDING, instruction="go"))

    runtime._stop_before_exit()
    assert publisher.payloads == [{"type": "stop"}]

    runtime._states.store(state(Lifecycle.IDLE))
    runtime._stop_before_exit()
    assert publisher.payloads == [{"type": "stop"}]


def test_heartbeat_worker_sends_immediately_and_reports_failure() -> None:
    runtime = CliRuntime(cli_config(), output_fn=lambda _message: None)

    class FailingPublisher:
        def put(self, _payload: str) -> None:
            raise RuntimeError("link failed")

    runtime._heartbeat_publisher = FailingPublisher()
    runtime._run_heartbeat()

    assert isinstance(runtime._heartbeat_error, RuntimeError)
