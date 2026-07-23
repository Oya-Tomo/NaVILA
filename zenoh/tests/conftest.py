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
            "zenoh_key_prefix": "navila",
            "node_state_publish_frequency_hz": 20.0,
            "go2_velocity_publish_frequency_hz": 20.0,
            "model": {"key": "model", "quantization": "4bit"},
            "camera": {
                "key": "camera/front",
                "sample_frequency_hz": 1.0,
                "frame_freshness_seconds": 2.5,
            },
            "inference": {"frequency_hz": 1.0},
            "motion": {
                "forward_velocity_mps": 0.5,
                "turn_velocity_rps": 1.0,
            },
            "control": {"cli_heartbeat_timeout_seconds": 1.0},
            "go2": {
                "zenoh_key_prefix": "unitree/go2",
                "node_state_timeout_seconds": 0.5,
                "stand_timeout_seconds": 10.0,
                "down_timeout_seconds": 10.0,
            },
        }
    )
