from __future__ import annotations

from pathlib import Path

import pytest
from config import (
    CliConfig,
    HeartbeatCommand,
    InstructionCommand,
    Lifecycle,
    NodeConfig,
    VelocityCommand,
    decode_control_command,
    decode_node_state,
    load_cli_config,
    load_node_config,
)
from pydantic import ValidationError


def test_example_configs_load() -> None:
    root = Path(__file__).resolve().parents[1]

    node = load_node_config(root / "node-config.example.json5")
    cli = load_cli_config(root / "cli-config.example.json5")

    assert node.camera.key == "camera/front"
    assert node.publish.velocity_frequency_hz == 20.0
    assert cli.heartbeat_interval_sec == 0.2


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("camera", "sample_frequency_hz"), 1),
        (("publish", "velocity_frequency_hz"), "20.0"),
        (("go2", "state_stale_timeout_sec"), False),
    ],
)
def test_node_config_rejects_non_strict_types(
    node_config: NodeConfig,
    path: tuple[str, str],
    value: object,
) -> None:
    document = node_config.dict()
    document[path[0]][path[1]] = value

    with pytest.raises(ValidationError):
        NodeConfig.parse_obj(document)


def test_config_rejects_unknown_fields_and_wildcard_keys(node_config: NodeConfig) -> None:
    document = node_config.dict()
    document["unexpected"] = True
    with pytest.raises(ValidationError):
        NodeConfig.parse_obj(document)

    document = node_config.dict()
    document["camera"]["key"] = "camera/**"
    with pytest.raises(ValidationError):
        NodeConfig.parse_obj(document)


def test_config_rejects_unsafe_ranges_and_timeout_relationships(node_config: NodeConfig) -> None:
    document = node_config.dict()
    document["motion"]["forward_velocity_mps"] = 4.0
    with pytest.raises(ValidationError):
        NodeConfig.parse_obj(document)

    document = node_config.dict()
    document["camera"]["stale_timeout_sec"] = 1.0
    with pytest.raises(ValidationError):
        NodeConfig.parse_obj(document)

    with pytest.raises(ValidationError):
        CliConfig.parse_obj(
            {
                "node_key": "navila",
                "heartbeat_interval_sec": 1.0,
                "node_state_stale_timeout_sec": 1.0,
                "command_timeout_sec": 15.0,
            }
        )


def test_json5_loader_rejects_duplicate_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "duplicate.json5"
    config_path.write_text("{node_key: 'a', node_key: 'b'}", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        load_node_config(config_path)


def test_control_wire_models_are_discriminated_and_strict() -> None:
    assert decode_control_command(b'{"type":"heartbeat"}') == HeartbeatCommand()
    assert decode_control_command(b'{"type":"instruction","instruction":"  go forward  "}') == InstructionCommand(
        instruction="go forward"
    )

    with pytest.raises(ValidationError):
        decode_control_command(b'{"type":"start","unexpected":true}')
    with pytest.raises(ValueError, match="unsupported"):
        decode_control_command(b'{"type":"launch"}')
    with pytest.raises(ValidationError):
        InstructionCommand(instruction="line one\nline two")
    with pytest.raises(ValidationError):
        InstructionCommand(instruction="\nline one")


def test_state_and_velocity_wire_models_reject_coercion_and_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        VelocityCommand(vx="0.5", vy=0.0, vyaw=0.0)
    with pytest.raises(ValidationError):
        VelocityCommand(vx=4.0, vy=0.0, vyaw=0.0)

    payload = b"""{
        "lifecycle":"idle",
        "cli_connected":true,
        "start_ready":false,
        "status_message":"waiting",
        "instruction":"",
        "inference":{"active":false,"last_duration_sec":null,"last_output":null},
        "action":null,
        "last_error":null
    }"""
    assert decode_node_state(payload).lifecycle is Lifecycle.IDLE

    with pytest.raises(ValidationError):
        decode_node_state(payload[:-2] + b',"unexpected":true}')
