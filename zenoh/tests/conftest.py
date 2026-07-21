from __future__ import annotations

import sys
from pathlib import Path

import pytest

ZENOH_SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ZENOH_SOURCE))

from config import NodeConfig  # noqa: E402


@pytest.fixture
def node_config() -> NodeConfig:
    return NodeConfig.parse_obj(
        {
            "node_key": "navila",
            "model": {"key": "model", "quantization": "4bit"},
            "camera": {
                "key": "camera/front",
                "sample_frequency_hz": 1.0,
                "stale_timeout_sec": 2.5,
            },
            "inference": {"frequency_hz": 1.0},
            "motion": {
                "forward_velocity_mps": 0.5,
                "turn_velocity_rps": 1.0,
            },
            "publish": {
                "velocity_frequency_hz": 20.0,
                "state_frequency_hz": 20.0,
            },
            "control": {"cli_heartbeat_timeout_sec": 1.0},
            "go2": {
                "robot_key": "unitree/go2",
                "state_stale_timeout_sec": 0.5,
                "stand_timeout_sec": 10.0,
                "down_timeout_sec": 10.0,
            },
        }
    )
