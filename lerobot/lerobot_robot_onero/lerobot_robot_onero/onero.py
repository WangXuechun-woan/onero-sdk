"""Onero A1 robot implementation for LeRobot.

Integrates the Onero SDK with LeRobot's Robot interface, enabling:
- Teleoperation (leader-follower)
- Data collection (record episodes)
- Policy inference (ACT, Diffusion Policy, etc.)
"""

from typing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots import Robot

from onero_sdk import Arm
from onero_sdk.protocols import ArmProtocol
from onero_sdk.model_registry import get_model

from .config_onero import OneroRobotConfig


class OneroRobot(Robot):
    """LeRobot-compatible driver for the Onero A1 arm.

    Usage:
        # Via LeRobot CLI:
        lerobot-teleoperate --robot.type=onero --robot.can_channel=can0

        # Via Python:
        from lerobot_robot_onero import OneroRobot, OneroRobotConfig
        robot = OneroRobot(OneroRobotConfig(can_channel="can0"))
        robot.connect()
        obs = robot.get_observation()
        robot.send_action(action)
        robot.disconnect()
    """

    config_class = OneroRobotConfig
    name = "onero"

    def __init__(self, config: OneroRobotConfig):
        super().__init__(config)
        self.config = config
        self._arm: ArmProtocol | None = None
        self._joint_specs = None
        self._joint_names: list[str] = []
        self.cameras = make_cameras_from_configs(config.cameras)

    # ─── Features ─────────────────────────────────────────────────────

    @property
    def _motors_ft(self) -> dict[str, type]:
        model = get_model(self.config.model)
        return {f"joint_{i+1}.pos": float for i in range(model.num_joints)}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam_key: (cam.height, cam.width, 3)
            for cam_key, cam in self.cameras.items()
        }

    @property
    def observation_features(self) -> dict:
        return {**self._motors_ft, **self._cameras_ft}

    @property
    def action_features(self) -> dict:
        return self._motors_ft

    # ─── Connection ───────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        if self._arm is None:
            return False
        if not self._arm.enabled:
            return False
        return all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        model = get_model(self.config.model)
        self._joint_specs = model.joints
        self._joint_names = [f"joint_{i+1}" for i in range(model.num_joints)]

        self._arm = Arm(self.config.model, channel=self.config.can_channel)
        self._arm.enable()

        for cam in self.cameras.values():
            cam.connect()

    def disconnect(self) -> None:
        if self._arm is not None:
            self._arm.disable()
            self._arm = None
        for cam in self.cameras.values():
            cam.disconnect()

    # ─── Calibration ──────────────────────────────────────────────────

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    # ─── I/O ──────────────────────────────────────────────────────────

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise ConnectionError("OneroRobot is not connected.")

        state = self._arm.state
        obs: dict[str, Any] = {}

        for i, name in enumerate(self._joint_names):
            obs[f"{name}.pos"] = self._normalize(state.joints[i], i)

        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.async_read()

        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        positions = []
        for i, name in enumerate(self._joint_names):
            normalized = action[f"{name}.pos"]
            positions.append(self._denormalize(normalized, i))

        self._arm.servoj(positions)
        return action

    # ─── Normalization ────────────────────────────────────────────────

    def _normalize(self, value: float, joint_idx: int) -> float:
        """Map physical position [p_min, p_max] → [-1, 1]."""
        spec = self._joint_specs[joint_idx]
        mid = (spec.p_max + spec.p_min) / 2.0
        half_range = (spec.p_max - spec.p_min) / 2.0
        if half_range < 1e-6:
            return 0.0
        return (value - mid) / half_range

    def _denormalize(self, value: float, joint_idx: int) -> float:
        """Map normalized [-1, 1] → physical position [p_min, p_max]."""
        spec = self._joint_specs[joint_idx]
        mid = (spec.p_max + spec.p_min) / 2.0
        half_range = (spec.p_max - spec.p_min) / 2.0
        pos = mid + value * half_range
        return max(spec.p_min, min(spec.p_max, pos))
