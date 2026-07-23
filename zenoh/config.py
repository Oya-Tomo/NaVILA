"""Validated JSON5 settings and wire models for the NaVILA Zenoh nodes."""

from __future__ import annotations

import json
import math
from enum import Enum
from pathlib import Path
from typing import Literal, Union

import json5
from pydantic import (
    BaseModel,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationError,
    confloat,
    root_validator,
    validator,
)

import zenoh

JSON_ENCODING = "application/json"
MAX_INSTRUCTION_LENGTH = 4096
GO2_VX_MIN = -2.5
GO2_VX_MAX = 3.8
GO2_VY_MAX = 1.0
GO2_VYAW_MAX = 4.0

PositiveFloat = confloat(strict=True, gt=0, allow_inf_nan=False)
NonNegativeFloat = confloat(strict=True, ge=0, allow_inf_nan=False)
FiniteFloat = confloat(strict=True, allow_inf_nan=False)
ForwardVelocity = confloat(strict=True, gt=0, le=GO2_VX_MAX, allow_inf_nan=False)
TurnVelocity = confloat(strict=True, gt=0, le=GO2_VYAW_MAX, allow_inf_nan=False)
Go2VelocityX = confloat(strict=True, ge=GO2_VX_MIN, le=GO2_VX_MAX, allow_inf_nan=False)
Go2VelocityY = confloat(strict=True, ge=-GO2_VY_MAX, le=GO2_VY_MAX, allow_inf_nan=False)
Go2VelocityYaw = confloat(strict=True, ge=-GO2_VYAW_MAX, le=GO2_VYAW_MAX, allow_inf_nan=False)


class FrozenModel(BaseModel):
    """Immutable Pydantic v1 model with an explicit wire/config surface."""

    class Config:
        allow_mutation = False
        extra = "forbid"


class ExternalModel(BaseModel):
    """Validated subset of an independently versioned external payload."""

    class Config:
        allow_mutation = False
        extra = "ignore"


def validate_concrete_key(value: str, field_name: str) -> str:
    value = value.strip().rstrip("/")
    if not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    try:
        zenoh.KeyExpr(value)
    except zenoh.ZError as error:
        raise ValueError(f"{field_name} must be a valid Zenoh key: {error}") from error
    if "*" in value:
        raise ValueError(f"{field_name} must be a concrete Zenoh key without wildcards")
    return value


def _validate_frequency(value: float, field_name: str) -> float:
    try:
        period = 1.0 / value
    except OverflowError:
        raise ValueError(f"{field_name} must produce a finite period") from None
    if not math.isfinite(period) or period <= 0:
        raise ValueError(f"{field_name} must produce a finite period")
    return value


class ModelConfig(FrozenModel):
    key: StrictStr
    quantization: Literal["4bit", "8bit", "fp16"]

    @validator("key")
    def validate_key(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("model.key must be a non-empty string")
        return value


class CameraConfig(FrozenModel):
    key: StrictStr
    sample_frequency_hz: PositiveFloat
    stale_timeout_sec: PositiveFloat

    @validator("key")
    def validate_key(cls, value: str) -> str:
        return validate_concrete_key(value, "camera.key")

    @validator("sample_frequency_hz")
    def validate_sample_frequency(cls, value: float) -> float:
        return _validate_frequency(value, "camera.sample_frequency_hz")

    @root_validator
    def validate_stale_window(cls, values: dict[str, object]) -> dict[str, object]:
        frequency = values.get("sample_frequency_hz")
        timeout = values.get("stale_timeout_sec")
        if isinstance(frequency, (int, float)) and isinstance(timeout, (int, float)):
            if timeout <= 1.0 / frequency:
                raise ValueError("camera.stale_timeout_sec must exceed one sampling period")
        return values


class InferenceConfig(FrozenModel):
    frequency_hz: PositiveFloat

    @validator("frequency_hz")
    def validate_frequency(cls, value: float) -> float:
        return _validate_frequency(value, "inference.frequency_hz")


class MotionConfig(FrozenModel):
    forward_velocity_mps: ForwardVelocity
    turn_velocity_rps: TurnVelocity


class PublishConfig(FrozenModel):
    velocity_frequency_hz: PositiveFloat
    state_frequency_hz: PositiveFloat

    @validator("velocity_frequency_hz", "state_frequency_hz")
    def validate_frequency(cls, value: float, field: object) -> float:
        return _validate_frequency(value, f"publish.{getattr(field, 'name', 'frequency')}")


class ControlConfig(FrozenModel):
    cli_heartbeat_timeout_sec: PositiveFloat


class Go2Config(FrozenModel):
    robot_key: StrictStr
    state_stale_timeout_sec: PositiveFloat
    stand_timeout_sec: PositiveFloat
    down_timeout_sec: PositiveFloat

    @validator("robot_key")
    def validate_key(cls, value: str) -> str:
        return validate_concrete_key(value, "go2.robot_key")


class NodeConfig(FrozenModel):
    node_key: StrictStr
    model: ModelConfig
    camera: CameraConfig
    inference: InferenceConfig
    motion: MotionConfig
    publish: PublishConfig
    control: ControlConfig
    go2: Go2Config

    @validator("node_key")
    def validate_key(cls, value: str) -> str:
        return validate_concrete_key(value, "node_key")

    @root_validator
    def validate_derived_keys(cls, values: dict[str, object]) -> dict[str, object]:
        node_key = values.get("node_key")
        go2 = values.get("go2")
        if isinstance(node_key, str):
            validate_concrete_key(f"{node_key}/command", "derived command key")
            validate_concrete_key(f"{node_key}/state", "derived state key")
        if isinstance(go2, Go2Config):
            validate_concrete_key(f"{go2.robot_key}/command", "derived Go2 command key")
            validate_concrete_key(f"{go2.robot_key}/state", "derived Go2 state key")
        return values


class CliConfig(FrozenModel):
    node_key: StrictStr
    heartbeat_interval_sec: PositiveFloat
    node_state_stale_timeout_sec: PositiveFloat
    command_timeout_sec: PositiveFloat

    @validator("node_key")
    def validate_key(cls, value: str) -> str:
        return validate_concrete_key(value, "node_key")

    @root_validator
    def validate_timeouts(cls, values: dict[str, object]) -> dict[str, object]:
        interval = values.get("heartbeat_interval_sec")
        stale = values.get("node_state_stale_timeout_sec")
        command = values.get("command_timeout_sec")
        if isinstance(interval, (int, float)) and isinstance(stale, (int, float)) and stale <= interval:
            raise ValueError("node_state_stale_timeout_sec must exceed heartbeat_interval_sec")
        if isinstance(stale, (int, float)) and isinstance(command, (int, float)) and command <= stale:
            raise ValueError("command_timeout_sec must exceed node_state_stale_timeout_sec")
        return values


class Lifecycle(str, Enum):
    INITIALIZING = "initializing"
    IDLE = "idle"
    STANDING = "standing"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class ActionType(str, Enum):
    STOP = "stop"
    FORWARD = "forward"
    LEFT = "left"
    RIGHT = "right"


class HeartbeatCommand(FrozenModel):
    type: Literal["heartbeat"] = "heartbeat"


class InstructionCommand(FrozenModel):
    type: Literal["instruction"] = "instruction"
    instruction: StrictStr

    @validator("instruction")
    def validate_instruction(cls, value: str) -> str:
        if "\n" in value or "\r" in value:
            raise ValueError("instruction must be a single line")
        value = value.strip()
        if len(value) > MAX_INSTRUCTION_LENGTH:
            raise ValueError(f"instruction must not exceed {MAX_INSTRUCTION_LENGTH} characters")
        return value


class StartCommand(FrozenModel):
    type: Literal["start"] = "start"


class StopCommand(FrozenModel):
    type: Literal["stop"] = "stop"


ControlCommand = Union[HeartbeatCommand, InstructionCommand, StartCommand, StopCommand]


class VelocityCommand(FrozenModel):
    type: Literal["velocity"] = "velocity"
    vx: Go2VelocityX
    vy: Go2VelocityY
    vyaw: Go2VelocityYaw


class PostureCommand(FrozenModel):
    type: Literal["posture"] = "posture"
    posture: Literal["stand", "down"]


class ParsedAction(FrozenModel):
    type: ActionType
    amount: StrictInt
    vx: FiniteFloat
    vy: FiniteFloat
    vyaw: FiniteFloat
    duration_sec: NonNegativeFloat


class InferenceState(FrozenModel):
    active: StrictBool
    last_duration_sec: NonNegativeFloat | None
    last_output: StrictStr | None


class ActionState(FrozenModel):
    type: ActionType
    vx: FiniteFloat
    vy: FiniteFloat
    vyaw: FiniteFloat
    remaining_sec: NonNegativeFloat


class NodeState(FrozenModel):
    lifecycle: Lifecycle
    cli_connected: StrictBool
    start_ready: StrictBool
    status_message: StrictStr | None
    instruction: StrictStr
    inference: InferenceState
    action: ActionState | None
    last_error: StrictStr | None


class Go2RobotState(ExternalModel):
    state: StrictStr


class Go2NodeState(ExternalModel):
    robot_connected: StrictBool
    robot_state: Go2RobotState
    accepting_commands: StrictBool


_CONTROL_MODELS: dict[str, type[FrozenModel]] = {
    "heartbeat": HeartbeatCommand,
    "instruction": InstructionCommand,
    "start": StartCommand,
    "stop": StopCommand,
}


def decode_control_command(payload: bytes | str) -> ControlCommand:
    document = json.loads(payload)
    if not isinstance(document, dict):
        raise ValueError("control command must be a JSON object")
    command_type = document.get("type")
    model = _CONTROL_MODELS.get(command_type) if isinstance(command_type, str) else None
    if model is None:
        raise ValueError(f"unsupported control command type: {command_type!r}")
    return model.parse_obj(document)  # type: ignore[return-value]


def decode_go2_state(payload: bytes | str) -> Go2NodeState:
    return Go2NodeState.parse_raw(payload)


def decode_node_state(payload: bytes | str) -> NodeState:
    return NodeState.parse_raw(payload)


def _load_json5(path: Path) -> object:
    try:
        return json5.loads(path.read_text(encoding="utf-8"), allow_duplicate_keys=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"{path}: invalid JSON5: {error}") from error


def load_node_config(path: Path) -> NodeConfig:
    try:
        return NodeConfig.parse_obj(_load_json5(path))
    except ValidationError as error:
        raise ValueError(f"{path}: invalid node configuration:\n{error}") from error


def load_cli_config(path: Path) -> CliConfig:
    try:
        return CliConfig.parse_obj(_load_json5(path))
    except ValidationError as error:
        raise ValueError(f"{path}: invalid CLI configuration:\n{error}") from error
