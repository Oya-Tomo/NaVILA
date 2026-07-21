from __future__ import annotations

import math

import pytest
from config import (
    ActionType,
    Go2NodeState,
    Go2RobotState,
    HeartbeatCommand,
    InstructionCommand,
    Lifecycle,
    NodeConfig,
    StartCommand,
    StopCommand,
)
from controller import ActionParseError, Controller, parse_navigation_action
from PIL import Image


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def go2(state: str, *, connected: bool = True, accepting: bool = True) -> Go2NodeState:
    return Go2NodeState(
        connected=connected,
        robot=Go2RobotState(state=state),
        accepting_commands=accepting,
    )


def make_running(
    config: NodeConfig,
    clock: FakeClock,
    *,
    num_video_frames: int = 4,
) -> Controller:
    controller = Controller(config, num_video_frames=num_video_frames, clock=clock)
    controller.initialize()
    controller.append_camera_frame(Image.new("RGB", (8, 8), "white"))
    controller.update_go2_state(go2("down"))
    controller.advance()
    controller.handle_command(HeartbeatCommand())
    assert controller.handle_command(InstructionCommand(instruction="go to the elevator"))
    assert controller.handle_command(StartCommand())
    assert controller.take_effects().posture == "stand"
    controller.update_go2_state(go2("ready_stand"))
    controller.advance()
    assert controller.snapshot().lifecycle is Lifecycle.RUNNING
    return controller


@pytest.mark.parametrize(
    ("output", "action_type", "amount", "velocity"),
    [
        ("The next action is stop", ActionType.STOP, 0, (0.0, 0.0, 0.0)),
        ("The next action is move forward", ActionType.FORWARD, 25, (0.5, 0.0, 0.0)),
        ("The next action is move forward 70 cm", ActionType.FORWARD, 75, (0.5, 0.0, 0.0)),
        ("The next action is turn left 30 degree", ActionType.LEFT, 30, (0.0, 0.0, 1.0)),
        ("The next action is turn right 44 degrees", ActionType.RIGHT, 45, (0.0, 0.0, -1.0)),
    ],
)
def test_action_parser_bounds_eval_compatible_outputs(
    node_config: NodeConfig,
    output: str,
    action_type: ActionType,
    amount: int,
    velocity: tuple[float, float, float],
) -> None:
    action = parse_navigation_action(output, node_config)

    assert action.type is action_type
    assert action.amount == amount
    assert (action.vx, action.vy, action.vyaw) == velocity
    if action_type is ActionType.FORWARD:
        assert action.duration_sec == pytest.approx((amount / 100.0) / 0.5)
    elif action_type in (ActionType.LEFT, ActionType.RIGHT):
        assert action.duration_sec == pytest.approx(math.radians(amount))


@pytest.mark.parametrize(
    "output",
    [
        "The next action is move backward 25 cm",
        "The next action is move left 25 cm",
        "No decision",
        "Move forward 25 cm, then turn left 15 degree",
        "The next action is move forward -25 cm",
        "The next action is move forward twenty cm",
        "The next action is stop for now",
    ],
)
def test_action_parser_rejects_unsupported_or_ambiguous_outputs(
    node_config: NodeConfig,
    output: str,
) -> None:
    with pytest.raises(ActionParseError):
        parse_navigation_action(output, node_config)


def test_camera_callback_sampling_is_bounded_fifo_without_catch_up(node_config: NodeConfig) -> None:
    clock = FakeClock()
    controller = Controller(node_config, num_video_frames=2, clock=clock)
    controller.initialize()

    red = Image.new("RGB", (4, 4), "red")
    green = Image.new("RGB", (4, 4), "green")
    blue = Image.new("RGB", (4, 4), "blue")
    assert controller.append_camera_frame(red)
    clock.now = 0.99
    assert not controller.camera_sample_due()
    assert not controller.append_camera_frame(green)
    clock.now = 3.0
    assert controller.camera_sample_due()
    assert controller.append_camera_frame(green)
    clock.now = 3.01
    assert not controller.append_camera_frame(blue)
    clock.now = 4.0
    assert controller.append_camera_frame(blue)

    controller.update_go2_state(go2("down"))
    controller.advance()
    controller.handle_command(HeartbeatCommand())
    controller.handle_command(InstructionCommand(instruction="test"))
    assert controller.handle_command(StartCommand())
    controller.take_effects()
    controller.update_go2_state(go2("ready_stand"))
    controller.advance()
    request = controller.begin_inference()

    assert request is not None
    assert len(request.frames) == 2
    assert request.frames[0].getpixel((0, 0)) == (0, 128, 0)
    assert request.frames[1].getpixel((0, 0)) == (0, 0, 255)


def test_start_waits_for_ready_stand_and_publishes_zero_until_first_result(node_config: NodeConfig) -> None:
    clock = FakeClock()
    controller = Controller(node_config, num_video_frames=4, clock=clock)
    controller.initialize()
    controller.handle_command(HeartbeatCommand())
    controller.handle_command(InstructionCommand(instruction="test"))
    controller.append_camera_frame(Image.new("RGB", (4, 4)))
    controller.update_go2_state(go2("down"))
    controller.advance()

    assert controller.handle_command(StartCommand())
    assert controller.snapshot().lifecycle is Lifecycle.STANDING
    assert controller.velocity_for_publish() is None
    assert controller.take_effects().posture == "stand"

    controller.update_go2_state(go2("ready_stand", accepting=False))
    controller.advance()
    assert controller.snapshot().lifecycle is Lifecycle.STANDING
    controller.update_go2_state(go2("ready_stand", accepting=True))
    controller.advance()
    assert controller.snapshot().lifecycle is Lifecycle.RUNNING
    assert controller.velocity_for_publish().dict() == {
        "type": "velocity",
        "vx": 0.0,
        "vy": 0.0,
        "vyaw": 0.0,
    }


def test_stop_discards_in_flight_result_and_waits_for_down(node_config: NodeConfig) -> None:
    clock = FakeClock()
    controller = make_running(node_config, clock)
    assert controller.begin_inference() is not None

    assert controller.handle_command(StopCommand())
    assert controller.snapshot().lifecycle is Lifecycle.STOPPING
    assert controller.velocity_for_publish() is None
    effects = controller.take_effects()
    assert effects.zero_velocity
    assert effects.posture == "down"

    assert not controller.complete_inference("The next action is move forward 50 cm", duration_sec=0.8)
    controller.update_go2_state(go2("down"))
    controller.advance()

    state = controller.snapshot()
    assert state.lifecycle is Lifecycle.IDLE
    assert state.action is None
    assert state.inference.last_output is None


def test_current_action_replaces_previous_result_and_expires_to_zero(node_config: NodeConfig) -> None:
    clock = FakeClock()
    controller = make_running(node_config, clock)

    assert controller.begin_inference() is not None
    assert controller.complete_inference("The next action is move forward 75 cm", duration_sec=0.1)
    assert controller.velocity_for_publish().vx == 0.5

    clock.now = 0.2
    assert controller.begin_inference() is not None
    assert controller.complete_inference("The next action is turn right 15 degree", duration_sec=0.1)
    velocity = controller.velocity_for_publish()
    assert velocity.vx == 0.0
    assert velocity.vyaw == -1.0

    clock.now += math.radians(15) + 0.01
    assert controller.velocity_for_publish().dict() == {
        "type": "velocity",
        "vx": 0.0,
        "vy": 0.0,
        "vyaw": 0.0,
    }


def test_down_confirmation_waits_for_in_flight_inference_past_posture_deadline(
    node_config: NodeConfig,
) -> None:
    clock = FakeClock()
    controller = make_running(node_config, clock)
    assert controller.begin_inference() is not None
    controller.handle_command(StopCommand())
    controller.take_effects()

    clock.now = 11.0
    controller.update_go2_state(go2("down"))
    controller.advance()
    assert controller.snapshot().lifecycle is Lifecycle.STOPPING

    assert not controller.complete_inference("The next action is move forward 25 cm", duration_sec=11.0)
    controller.advance()
    assert controller.snapshot().lifecycle is Lifecycle.IDLE


def test_down_timeout_enters_error_and_remains_non_moving(node_config: NodeConfig) -> None:
    clock = FakeClock()
    controller = make_running(node_config, clock)
    controller.handle_command(StopCommand())
    controller.take_effects()

    clock.now = 10.0
    controller.update_go2_state(go2("ready_stand"))
    controller.advance()

    state = controller.snapshot()
    assert state.lifecycle is Lifecycle.ERROR
    assert state.action is None
    assert "down" in state.last_error
    assert controller.velocity_for_publish() is None


def test_stale_inputs_use_the_same_stop_path(node_config: NodeConfig) -> None:
    clock = FakeClock()
    controller = make_running(node_config, clock)

    clock.now = 0.51
    controller.handle_command(HeartbeatCommand())
    controller.append_camera_frame(Image.new("RGB", (4, 4)), sampled_at=0.51)
    controller.advance()

    state = controller.snapshot()
    effects = controller.take_effects()
    assert state.lifecycle is Lifecycle.ERROR
    assert "Go2 state" in state.last_error
    assert effects.zero_velocity
    assert effects.posture is None


def test_heartbeat_and_camera_staleness_request_safe_down(node_config: NodeConfig) -> None:
    clock = FakeClock()
    controller = make_running(node_config, clock)

    clock.now = 1.01
    controller.append_camera_frame(Image.new("RGB", (4, 4)))
    controller.update_go2_state(go2("ready_stand"))
    controller.advance()
    assert controller.snapshot().lifecycle is Lifecycle.STOPPING
    assert controller.take_effects().posture == "down"

    clock = FakeClock()
    controller = make_running(node_config, clock)
    clock.now = 2.51
    controller.handle_command(HeartbeatCommand())
    controller.update_go2_state(go2("ready_stand"))
    controller.advance()
    state = controller.snapshot()
    assert state.lifecycle is Lifecycle.STOPPING
    assert state.last_error is None
    assert controller.take_effects().posture == "down"


def test_startup_requests_down_once_without_cli(node_config: NodeConfig) -> None:
    clock = FakeClock()
    controller = Controller(node_config, num_video_frames=4, clock=clock)
    controller.initialize()
    controller.update_go2_state(go2("ready_stand", accepting=False))

    controller.advance()
    first = controller.take_effects()
    controller.advance()
    second = controller.take_effects()

    assert first.zero_velocity
    assert first.posture == "down"
    assert second.posture is None
