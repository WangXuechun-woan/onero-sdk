"""Onero Robot configuration for LeRobot."""

from dataclasses import dataclass, field
from typing import Any

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("onero")
@dataclass
class OneroRobotConfig(RobotConfig):
    """Configuration for Onero A1 robot arm.

    Args:
        can_channel: SocketCAN interface name (e.g. "can0").
        model: Arm model name (e.g. "A1R", "A1L").
        cameras: Camera configurations for observation.
    """

    can_channel: str = "can0"
    model: str = "A1R"
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
