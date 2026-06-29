"""Onero A1 teleoperator for LeRobot.

Uses a second A1 arm in freedrive (teach) mode as a master device.
The operator physically guides the leader arm, and joint positions
are read as actions for the follower robot.
"""

from typing import Any

from lerobot.teleoperators import Teleoperator

from onero_sdk import Arm
from onero_sdk.model_registry import get_model

from .config_onero_teleop import OneroTeleopConfig


class OneroTeleop(Teleoperator):
    """LeRobot teleoperator using an Onero A1 leader arm.

    Usage:
        lerobot-teleoperate --robot.type=onero --robot.can_channel=can0 \\
            --teleop.type=onero_teleop --teleop.leader_channel=can1
    """

    config_class = OneroTeleopConfig
    name = "onero_teleop"

    def __init__(self, config: OneroTeleopConfig):
        super().__init__(config)
        self.config = config
        self._leader: Arm | None = None
        self._joint_specs = None
        self._joint_names: list[str] = []

    @property
    def _motors_ft(self) -> dict[str, type]:
        model = get_model(self.config.model)
        return {f"joint_{i+1}.pos": float for i in range(model.num_joints)}

    @property
    def action_features(self) -> dict:
        return self._motors_ft

    @property
    def feedback_features(self) -> dict:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self._leader is not None and self._leader.enabled

    def connect(self, calibrate: bool = True) -> None:
        model = get_model(self.config.model)
        self._joint_specs = model.joints
        self._joint_names = [f"joint_{i+1}" for i in range(model.num_joints)]

        self._leader = Arm(self.config.model, channel=self.config.leader_channel)
        self._leader.enable()
        self._leader.freedrive_enable(damping=self.config.damping)

    def disconnect(self) -> None:
        if self._leader is not None:
            self._leader.freedrive_disable()
            self._leader.disable()
            self._leader = None

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        """No separate LeRobot calibration artifact is required for this teleop device."""
        return None

    def configure(self) -> None:
        """Runtime configuration is supplied through OneroTeleopConfig."""
        return None

    def get_action(self) -> dict[str, Any]:
        """Read leader arm joint positions as normalized action."""
        state = self._leader.state
        return {
            f"{name}.pos": self._normalize(state.joints[i], i)
            for i, name in enumerate(self._joint_names)
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """No force feedback on leader arm (passive freedrive)."""
        return None

    def _normalize(self, value: float, joint_idx: int) -> float:
        spec = self._joint_specs[joint_idx]
        mid = (spec.p_max + spec.p_min) / 2.0
        half_range = (spec.p_max - spec.p_min) / 2.0
        if half_range < 1e-6:
            return 0.0
        return (value - mid) / half_range
