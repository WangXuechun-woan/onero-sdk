"""Onero teleoperator configuration for LeRobot."""

from dataclasses import dataclass

from lerobot.teleoperators import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("onero_teleop")
@dataclass
class OneroTeleopConfig(TeleoperatorConfig):
    """Configuration for Onero A1 as teleoperation leader arm.

    Uses a second A1 arm in freedrive mode as a master (leader) device.
    The operator physically moves the leader arm, and its joint angles
    are sent as actions to the follower robot.

    Args:
        leader_channel: CAN channel for the leader arm.
        model: Arm model name.
        damping: Freedrive damping (resistance when guiding).
    """

    leader_channel: str = "can1"
    model: str = "A1"
    damping: float = 0.3
