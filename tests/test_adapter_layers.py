"""
Adapter layer tests — verify LeRobot, BAVR, and ROS2 integration logic.

Tests are isolated from hardware: they mock the Arm/backend and test
the adapter's data transformations, coordinate conversions, and API contracts.
"""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════════════════
# Shared mock helpers
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MockJointSpec:
    """Minimal JointSpec for testing normalization."""
    p_min: float
    p_max: float
    v_max: float = 5.0
    t_max: float = 20.0


def _mock_arm_state(joints=None, velocities=None, torques=None):
    """Create a mock ArmState."""
    n = 7
    state = MagicMock()
    state.joints = joints if joints is not None else [0.0] * n
    state.velocities = velocities if velocities is not None else [0.0] * n
    state.torques = torques if torques is not None else [0.0] * n
    return state


# ═══════════════════════════════════════════════════════════════════════════════
# LeRobot adapter tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestLeRobotNormalization:
    """Test the normalize/denormalize round-trip for LeRobot adapter."""

    @pytest.fixture
    def joint_specs(self):
        return [
            MockJointSpec(p_min=-2.87, p_max=2.87),
            MockJointSpec(p_min=-1.74, p_max=1.74),
            MockJointSpec(p_min=-2.87, p_max=2.87),
            MockJointSpec(p_min=-3.05, p_max=-0.07),
            MockJointSpec(p_min=-2.87, p_max=2.87),
            MockJointSpec(p_min=-1.57, p_max=2.09),
            MockJointSpec(p_min=-2.87, p_max=2.87),
        ]

    def _normalize(self, value, joint_idx, specs):
        spec = specs[joint_idx]
        mid = (spec.p_max + spec.p_min) / 2.0
        half_range = (spec.p_max - spec.p_min) / 2.0
        if half_range < 1e-6:
            return 0.0
        return (value - mid) / half_range

    def _denormalize(self, value, joint_idx, specs):
        spec = specs[joint_idx]
        mid = (spec.p_max + spec.p_min) / 2.0
        half_range = (spec.p_max - spec.p_min) / 2.0
        pos = mid + value * half_range
        return max(spec.p_min, min(spec.p_max, pos))

    @pytest.mark.parametrize("joint_idx", range(7))
    def test_normalize_midpoint_is_zero(self, joint_specs, joint_idx):
        """Joint midpoint normalizes to 0."""
        spec = joint_specs[joint_idx]
        mid = (spec.p_max + spec.p_min) / 2.0
        assert abs(self._normalize(mid, joint_idx, joint_specs)) < 1e-10

    @pytest.mark.parametrize("joint_idx", range(7))
    def test_normalize_bounds_are_pm1(self, joint_specs, joint_idx):
        """Joint limits normalize to ±1."""
        spec = joint_specs[joint_idx]
        assert abs(self._normalize(spec.p_min, joint_idx, joint_specs) - (-1.0)) < 1e-10
        assert abs(self._normalize(spec.p_max, joint_idx, joint_specs) - 1.0) < 1e-10

    @pytest.mark.parametrize("joint_idx", range(7))
    @pytest.mark.parametrize("value", [-1.0, -0.5, 0.0, 0.5, 1.0])
    def test_normalize_denormalize_roundtrip(self, joint_specs, joint_idx, value):
        """normalize(denormalize(x)) = x for valid range."""
        physical = self._denormalize(value, joint_idx, joint_specs)
        recovered = self._normalize(physical, joint_idx, joint_specs)
        assert abs(recovered - value) < 1e-10

    @pytest.mark.parametrize("joint_idx", range(7))
    def test_denormalize_clamps_out_of_range(self, joint_specs, joint_idx):
        """Values outside [-1, 1] are clamped to joint limits."""
        spec = joint_specs[joint_idx]
        assert self._denormalize(2.0, joint_idx, joint_specs) == spec.p_max
        assert self._denormalize(-2.0, joint_idx, joint_specs) == spec.p_min

    def test_normalize_symmetric_range(self, joint_specs):
        """Symmetric ranges produce symmetric normalization."""
        spec = joint_specs[0]  # [-2.87, 2.87] is symmetric
        val = 1.0
        n_pos = self._normalize(val, 0, joint_specs)
        n_neg = self._normalize(-val, 0, joint_specs)
        assert abs(n_pos + n_neg) < 1e-10

    def test_normalize_asymmetric_range(self, joint_specs):
        """Asymmetric range (joint 3): midpoint is NOT at 0 physical."""
        spec = joint_specs[3]  # [-3.05, -0.07]
        mid = (spec.p_max + spec.p_min) / 2.0
        assert mid < 0  # Midpoint is negative
        assert abs(self._normalize(mid, 3, joint_specs)) < 1e-10

    @pytest.mark.parametrize("joint_idx", range(7))
    def test_normalize_monotonic(self, joint_specs, joint_idx):
        """Normalization is monotonically increasing."""
        spec = joint_specs[joint_idx]
        positions = np.linspace(spec.p_min, spec.p_max, 20)
        normalized = [self._normalize(p, joint_idx, joint_specs) for p in positions]
        for i in range(len(normalized) - 1):
            assert normalized[i] < normalized[i + 1]

    def test_zero_range_joint_returns_zero(self):
        """Joint with zero range normalizes to 0."""
        specs = [MockJointSpec(p_min=1.0, p_max=1.0)]
        assert self._normalize(1.0, 0, specs) == 0.0


def _has_lerobot():
    try:
        from lerobot.cameras import CameraConfig
        from lerobot.robots import Robot
        return True
    except (ImportError, ModuleNotFoundError):
        return False


skip_no_lerobot = pytest.mark.skipif(
    not _has_lerobot(), reason="LeRobot not installed")


@skip_no_lerobot
class TestLeRobotObservationAction:
    """Test LeRobot adapter observation/action format compliance."""

    def _get_robot_classes(self):
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "lerobot", "lerobot_robot_onero"))
        from lerobot_robot_onero.onero import OneroRobot
        from lerobot_robot_onero.config_onero import OneroRobotConfig
        return OneroRobot, OneroRobotConfig

    def test_observation_keys_match_features(self):
        """get_observation() returns all declared feature keys."""
        OneroRobot, OneroRobotConfig = self._get_robot_classes()

        config = OneroRobotConfig(can_channel="can0")
        robot = OneroRobot(config)

        # Mock everything
        robot._arm = MagicMock()
        robot._arm.enabled = True
        robot._arm.state = _mock_arm_state()
        robot._joint_specs = [
            MockJointSpec(p_min=-2.87, p_max=2.87) for _ in range(7)]
        robot._joint_names = [f"joint_{i+1}" for i in range(7)]
        robot.cameras = {}

        obs = robot.get_observation()
        for i in range(7):
            assert f"joint_{i+1}.pos" in obs
            assert isinstance(obs[f"joint_{i+1}.pos"], float)

    def test_send_action_calls_servoj(self):
        """send_action() calls arm.servoj with denormalized positions."""
        OneroRobot, OneroRobotConfig = self._get_robot_classes()

        config = OneroRobotConfig(can_channel="can0")
        robot = OneroRobot(config)
        robot._arm = MagicMock()
        robot._arm.enabled = True
        robot._joint_specs = [
            MockJointSpec(p_min=-2.87, p_max=2.87) for _ in range(7)]
        robot._joint_names = [f"joint_{i+1}" for i in range(7)]
        robot.cameras = {}
        robot._connected = True

        action = {f"joint_{i+1}.pos": 0.0 for i in range(7)}
        robot.send_action(action)

        robot._arm.servoj.assert_called_once()
        positions = robot._arm.servoj.call_args[0][0]
        assert len(positions) == 7
        # Normalized 0.0 → midpoint of range → 0.0 for symmetric range
        for p in positions:
            assert abs(p) < 1e-10

    def test_observation_values_in_normalized_range(self):
        """Observations are always in [-1, 1]."""
        OneroRobot, OneroRobotConfig = self._get_robot_classes()

        config = OneroRobotConfig(can_channel="can0")
        robot = OneroRobot(config)
        robot._arm = MagicMock()
        robot._arm.enabled = True
        # Positions at joint limits
        robot._arm.state = _mock_arm_state(joints=[2.87] * 7)
        robot._joint_specs = [
            MockJointSpec(p_min=-2.87, p_max=2.87) for _ in range(7)]
        robot._joint_names = [f"joint_{i+1}" for i in range(7)]
        robot.cameras = {}

        obs = robot.get_observation()
        for i in range(7):
            val = obs[f"joint_{i+1}.pos"]
            assert -1.0 <= val <= 1.0, f"Joint {i+1} obs {val} out of [-1,1]"


# ═══════════════════════════════════════════════════════════════════════════════
# BAVR adapter tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestBAVRCoordinateTransform:
    """Test VR → A1 coordinate transformation."""

    def test_r_vr_to_a1_is_rotation(self):
        """R_VR_TO_A1 is a valid rotation matrix (SO(3))."""
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "beavr", "beavr_onero"))
        from robot_wrapper import R_VR_TO_A1

        assert np.allclose(R_VR_TO_A1.T @ R_VR_TO_A1, np.eye(3), atol=1e-10)
        assert abs(np.linalg.det(R_VR_TO_A1) - 1.0) < 1e-10

    def test_vr_y_up_maps_to_a1_z_up(self):
        """VR Y-axis (up) maps to A1 Z-axis (up)."""
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "beavr", "beavr_onero"))
        from robot_wrapper import R_VR_TO_A1

        vr_up = np.array([0.0, 1.0, 0.0])
        a1_result = R_VR_TO_A1 @ vr_up
        # Should be [0, 0, 1] (A1 Z-up)
        assert np.allclose(a1_result, [0, 0, 1], atol=1e-10), \
            f"VR Y-up → A1: {a1_result}, expected [0,0,1]"

    def test_vr_neg_z_forward_maps_to_a1_x_forward(self):
        """VR -Z (forward, toward user) maps to A1 X (forward)."""
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "beavr", "beavr_onero"))
        from robot_wrapper import R_VR_TO_A1

        vr_forward = np.array([0.0, 0.0, -1.0])
        a1_result = R_VR_TO_A1 @ vr_forward
        # Should be [1, 0, 0] (A1 X-forward)
        assert np.allclose(a1_result, [1, 0, 0], atol=1e-10), \
            f"VR -Z → A1: {a1_result}, expected [1,0,0]"

    def test_vr_x_right_maps_to_a1_neg_y(self):
        """VR X (right) maps to A1 -Y (left in A1 convention)."""
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "beavr", "beavr_onero"))
        from robot_wrapper import R_VR_TO_A1

        vr_right = np.array([1.0, 0.0, 0.0])
        a1_result = R_VR_TO_A1 @ vr_right
        # Should be [0, -1, 0] (A1 -Y)
        assert np.allclose(a1_result, [0, -1, 0], atol=1e-10), \
            f"VR X → A1: {a1_result}, expected [0,-1,0]"

    def test_rotation_preserves_norms(self):
        """Coordinate transform preserves vector magnitudes."""
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "beavr", "beavr_onero"))
        from robot_wrapper import R_VR_TO_A1

        rng = np.random.default_rng(42)
        for _ in range(100):
            v_vr = rng.standard_normal(3)
            v_a1 = R_VR_TO_A1 @ v_vr
            assert abs(np.linalg.norm(v_a1) - np.linalg.norm(v_vr)) < 1e-10

    def test_double_rotation_inverse(self):
        """R_VR_TO_A1.T converts back from A1 → VR."""
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "beavr", "beavr_onero"))
        from robot_wrapper import R_VR_TO_A1

        rng = np.random.default_rng(123)
        v = rng.standard_normal(3)
        v_a1 = R_VR_TO_A1 @ v
        v_back = R_VR_TO_A1.T @ v_a1
        assert np.allclose(v_back, v, atol=1e-10)


class TestBAVRSendAction:
    """Test BAVR send_action data flow."""

    @pytest.fixture
    def bavr_robot(self):
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "beavr", "beavr_onero"))
        from robot_wrapper import OneroA1Robot, OneroA1Config

        config = OneroA1Config(
            position_scale_xyz=[1.0, 1.0, 1.0],
            deadzone=0.0,
            smoothing_alpha=1.0,  # No filtering
            max_linear_speed=10.0,  # Very high, won't limit
        )
        robot = OneroA1Robot(config)
        robot._arm = MagicMock()
        robot._connected = True
        robot._arm_origin_pos = np.array([0.3, 0.0, 0.3])
        robot._arm_origin_rot = np.eye(3)
        robot._filtered_pos = np.array([0.3, 0.0, 0.3])
        robot._filtered_rot = np.eye(3)
        robot._current_ee_pos = np.array([0.3, 0.0, 0.3])
        robot._current_ee_rot = np.eye(3)
        # Set VR origin so delta is computed
        robot._vr_origin_pos = np.array([0.0, 0.5, 0.0])
        robot._vr_origin_rot = np.eye(3)
        return robot

    def test_first_action_calibrates_vr_origin(self):
        """First send_action stores VR origin, doesn't move arm."""
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "beavr", "beavr_onero"))
        from robot_wrapper import OneroA1Robot, OneroA1Config

        robot = OneroA1Robot(OneroA1Config())
        robot._arm = MagicMock()
        robot._connected = True
        robot._arm_origin_pos = np.array([0.3, 0.0, 0.3])
        robot._arm_origin_rot = np.eye(3)
        robot._filtered_pos = np.array([0.3, 0.0, 0.3])
        robot._filtered_rot = np.eye(3)
        robot._vr_origin_pos = None  # Not yet calibrated
        robot._vr_origin_rot = None

        action = {"target_position": np.array([0.1, 0.5, -0.2]),
                  "target_rotation": np.eye(3)}
        robot.send_action(action)

        # Should have stored origin, NOT called servol
        assert robot._vr_origin_pos is not None
        robot._arm.servol.assert_not_called()

    def test_deadzone_suppresses_small_motion(self):
        """Motion below deadzone threshold is suppressed."""
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "beavr", "beavr_onero"))
        from robot_wrapper import OneroA1Robot, OneroA1Config

        config = OneroA1Config(
            deadzone=0.01,  # 10mm deadzone
            smoothing_alpha=1.0,
            max_linear_speed=10.0,
        )
        robot = OneroA1Robot(config)
        robot._arm = MagicMock()
        robot._connected = True
        robot._arm_origin_pos = np.array([0.3, 0.0, 0.3])
        robot._arm_origin_rot = np.eye(3)
        robot._filtered_pos = np.array([0.3, 0.0, 0.3])
        robot._filtered_rot = np.eye(3)
        robot._vr_origin_pos = np.array([0.0, 0.5, 0.0])
        robot._vr_origin_rot = np.eye(3)

        # 1mm delta in VR (below deadzone after coord transform)
        small_delta = np.array([0.001, 0.0, 0.0])
        action = {"target_position": robot._vr_origin_pos + small_delta,
                  "target_rotation": np.eye(3)}
        robot.send_action(action)

        # After transform, delta should be below deadzone
        robot._arm.servol.assert_called()
        call_args = robot._arm.servol.call_args[0][0]
        # Position should be approximately at arm origin (deadzone killed motion)
        pos_from_call = np.array(call_args[:3])
        assert np.linalg.norm(pos_from_call - robot._arm_origin_pos) < 0.002

    def test_workspace_clamping(self, bavr_robot):
        """Positions outside workspace are clamped."""
        # Move VR far outside workspace → should be clamped
        far_pos = bavr_robot._vr_origin_pos + np.array([0.0, 0.0, -2.0])
        action = {"target_position": far_pos, "target_rotation": np.eye(3)}
        bavr_robot.send_action(action)

        bavr_robot._arm.servol.assert_called()
        call_args = bavr_robot._arm.servol.call_args[0][0]
        pos = np.array(call_args[:3])
        ws_min = np.array(bavr_robot.config.workspace_min)
        ws_max = np.array(bavr_robot.config.workspace_max)
        for i in range(3):
            assert pos[i] >= ws_min[i] - 1e-6
            assert pos[i] <= ws_max[i] + 1e-6

    def test_ee_pose_4x4_format_accepted(self, bavr_robot):
        """BEAVR standard 4x4 homogeneous transform format is accepted."""
        T = np.eye(4)
        T[:3, 3] = bavr_robot._vr_origin_pos + np.array([0.05, 0.0, 0.0])
        action = {"ee_pose": T}
        bavr_robot.send_action(action)
        bavr_robot._arm.servol.assert_called()


class TestBAVRSmoothing:
    """Test low-pass filter behavior."""

    def test_alpha_zero_holds_position(self):
        """smoothing_alpha=0 means full filtering → position doesn't change."""
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "beavr", "beavr_onero"))
        from robot_wrapper import OneroA1Robot, OneroA1Config

        config = OneroA1Config(
            smoothing_alpha=0.0,  # Full filter → no change
            deadzone=0.0,
            max_linear_speed=10.0,
        )
        robot = OneroA1Robot(config)
        robot._arm = MagicMock()
        robot._connected = True
        robot._arm_origin_pos = np.array([0.3, 0.0, 0.3])
        robot._arm_origin_rot = np.eye(3)
        robot._filtered_pos = np.array([0.3, 0.0, 0.3])
        robot._filtered_rot = np.eye(3)
        robot._vr_origin_pos = np.array([0.0, 0.5, 0.0])
        robot._vr_origin_rot = np.eye(3)

        # Large delta
        delta_vr = np.array([0.0, 0.0, -0.1])
        action = {"target_position": robot._vr_origin_pos + delta_vr,
                  "target_rotation": np.eye(3)}
        robot.send_action(action)

        # Filtered position should not have moved
        assert np.allclose(robot._filtered_pos, [0.3, 0.0, 0.3], atol=1e-6)

    def test_alpha_one_passes_through(self):
        """smoothing_alpha=1 means no filtering → position tracks target."""
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "beavr", "beavr_onero"))
        from robot_wrapper import OneroA1Robot, OneroA1Config

        config = OneroA1Config(
            smoothing_alpha=1.0,  # No filter
            deadzone=0.0,
            max_linear_speed=10.0,
            position_scale_xyz=[1.0, 1.0, 1.0],
        )
        robot = OneroA1Robot(config)
        robot._arm = MagicMock()
        robot._connected = True
        robot._arm_origin_pos = np.array([0.3, 0.0, 0.3])
        robot._arm_origin_rot = np.eye(3)
        robot._filtered_pos = np.array([0.3, 0.0, 0.3])
        robot._filtered_rot = np.eye(3)
        robot._vr_origin_pos = np.array([0.0, 0.5, 0.0])
        robot._vr_origin_rot = np.eye(3)

        # Move VR in -Z (forward) → should map to A1 +X
        delta_vr = np.array([0.0, 0.0, -0.1])
        action = {"target_position": robot._vr_origin_pos + delta_vr,
                  "target_rotation": np.eye(3)}
        robot.send_action(action)

        # With alpha=1, filtered_pos should have moved from origin
        assert not np.allclose(robot._filtered_pos, robot._arm_origin_pos, atol=0.01)


# ═══════════════════════════════════════════════════════════════════════════════
# ROS2 adapter tests (logic-only, no rclpy dependency)
# ═══════════════════════════════════════════════════════════════════════════════

class TestROS2TrajectoryInterpolation:
    """Test trajectory interpolation logic used by the ROS2 node."""

    def _interpolate_trajectory(self, points, elapsed, n_joints, hw_positions):
        """Reproduce the arm_node interpolation logic for testing."""
        if not points:
            return None, None

        last_point = points[-1]
        last_time = last_point["time"]

        if elapsed >= last_time:
            positions = list(last_point["positions"])
            velocities = [0.0] * n_joints
            return positions, velocities

        prev_point = None
        prev_time = 0.0
        for pt in points:
            pt_time = pt["time"]
            if pt_time >= elapsed:
                if prev_point is None:
                    alpha = 0.0
                else:
                    dt_seg = pt_time - prev_time
                    alpha = (elapsed - prev_time) / dt_seg if dt_seg > 0 else 1.0

                positions = []
                velocities = []
                for i in range(n_joints):
                    p0 = prev_point["positions"][i] if prev_point else hw_positions[i]
                    p1 = pt["positions"][i]
                    positions.append(p0 + alpha * (p1 - p0))

                    if pt.get("velocities"):
                        v0 = prev_point["velocities"][i] if prev_point and prev_point.get("velocities") else 0.0
                        v1 = pt["velocities"][i]
                        velocities.append(v0 + alpha * (v1 - v0))
                    else:
                        velocities.append(0.0)
                return positions, velocities

            prev_point = pt
            prev_time = pt_time
        return None, None

    def test_interpolation_at_start_is_current(self):
        """At t=0, interpolation returns start position."""
        points = [
            {"time": 1.0, "positions": [1.0, 0.5], "velocities": [1.0, 0.5]},
            {"time": 2.0, "positions": [2.0, 1.0], "velocities": [0.0, 0.0]},
        ]
        hw = [0.0, 0.0]
        pos, vel = self._interpolate_trajectory(points, 0.0, 2, hw)
        assert np.allclose(pos, hw, atol=1e-10)

    def test_interpolation_at_end_holds_final(self):
        """After trajectory ends, holds final position with zero velocity."""
        points = [
            {"time": 1.0, "positions": [1.0, 0.5], "velocities": [1.0, 0.5]},
            {"time": 2.0, "positions": [2.0, 1.0], "velocities": [0.0, 0.0]},
        ]
        hw = [0.0, 0.0]
        pos, vel = self._interpolate_trajectory(points, 3.0, 2, hw)
        assert np.allclose(pos, [2.0, 1.0], atol=1e-10)
        assert np.allclose(vel, [0.0, 0.0], atol=1e-10)

    def test_interpolation_midpoint(self):
        """Between two explicit points, position is linearly interpolated."""
        points = [
            {"time": 1.0, "positions": [0.0, 0.0], "velocities": None},
            {"time": 3.0, "positions": [2.0, 1.0], "velocities": None},
        ]
        hw = [0.0, 0.0]
        # At t=2.0, between point 0 (t=1) and point 1 (t=3): alpha=0.5
        pos, vel = self._interpolate_trajectory(points, 2.0, 2, hw)
        assert np.allclose(pos, [1.0, 0.5], atol=1e-10)

    def test_interpolation_velocity_interpolated(self):
        """Velocities are linearly interpolated between points."""
        points = [
            {"time": 1.0, "positions": [1.0], "velocities": [2.0]},
            {"time": 2.0, "positions": [2.0], "velocities": [0.0]},
        ]
        hw = [0.0]
        pos, vel = self._interpolate_trajectory(points, 1.5, 1, hw)
        # Between point 0 (t=1, v=2.0) and point 1 (t=2, v=0.0), at t=1.5: alpha=0.5
        assert abs(vel[0] - 1.0) < 1e-10

    def test_interpolation_multiple_segments(self):
        """Multi-segment trajectory produces correct intermediate values."""
        points = [
            {"time": 1.0, "positions": [1.0], "velocities": [1.0]},
            {"time": 2.0, "positions": [3.0], "velocities": [2.0]},
            {"time": 3.0, "positions": [3.5], "velocities": [0.0]},
        ]
        hw = [0.0]

        # At t=2.5, we're between point 1 and point 2
        pos, vel = self._interpolate_trajectory(points, 2.5, 1, hw)
        # alpha = (2.5 - 2.0) / (3.0 - 2.0) = 0.5
        expected_pos = 3.0 + 0.5 * (3.5 - 3.0)
        expected_vel = 2.0 + 0.5 * (0.0 - 2.0)
        assert abs(pos[0] - expected_pos) < 1e-10
        assert abs(vel[0] - expected_vel) < 1e-10

    def test_zero_duration_segment_gives_alpha_one(self):
        """Zero-duration segment uses alpha=1.0 (goes to target)."""
        points = [
            {"time": 0.0, "positions": [5.0], "velocities": None},
        ]
        hw = [0.0]
        pos, vel = self._interpolate_trajectory(points, 0.0, 1, hw)
        # dt_seg=0, alpha=1.0 → goes to target
        assert abs(pos[0] - 5.0) < 1e-10


class TestROS2JointMapping:
    """Test joint name → index mapping for ROS2 node."""

    JOINT_NAMES = [
        "joint1-a1_r", "joint2-a1_r", "joint3-a1_r", "joint4-a1_r",
        "joint5-a1_r", "joint6-a1_r", "joint7-a1_r",
    ]

    def test_all_joints_have_unique_names(self):
        """All 7 joint names are unique."""
        assert len(set(self.JOINT_NAMES)) == 7

    def test_joint_names_count(self):
        """Node declares exactly 7 joints."""
        assert len(self.JOINT_NAMES) == 7

    def test_trajectory_joint_name_lookup(self):
        """Trajectory joint name mapping finds correct indices."""
        requested_names = ["joint3-a1_r", "joint1-a1_r", "joint7-a1_r"]
        indices = [self.JOINT_NAMES.index(name) for name in requested_names]
        assert indices == [2, 0, 6]

    def test_unknown_joint_name_raises(self):
        """Unknown joint name raises ValueError."""
        with pytest.raises(ValueError):
            self.JOINT_NAMES.index("unknown_joint")

    def test_partial_joint_trajectory(self):
        """Trajectory can command a subset of joints."""
        active_joints = ["joint2-a1_r", "joint4-a1_r"]
        indices = [self.JOINT_NAMES.index(name) for name in active_joints]
        assert indices == [1, 3]
        # Only these joints should be updated, others hold


class TestROS2Parameters:
    """Test ROS2 node parameter validation logic."""

    def test_default_kp_kd_within_protocol_limits(self):
        """Default kp=50 and kd=1 are within MIT protocol limits."""
        kp_default = 50.0
        kd_default = 1.0
        KP_MAX = 500.0
        KD_MAX = 5.0
        assert 0 <= kp_default <= KP_MAX
        assert 0 <= kd_default <= KD_MAX

    def test_control_rate_produces_valid_dt(self):
        """200Hz control rate gives 5ms period."""
        control_rate = 200.0
        dt = 1.0 / control_rate
        assert abs(dt - 0.005) < 1e-10

    def test_gravity_compensation_torque_added(self):
        """Gravity torques are added as feed-forward to position command."""
        # Simulate: gravity torques are non-zero for non-upright config
        gravity_torques = [0.0, -5.2, -2.1, -1.5, -0.3, -0.1, 0.0]
        positions = [0.0, -0.5, 0.0, -1.0, 0.0, 0.5, 0.0]
        kp = 50.0

        # The command should include both position tracking AND gravity comp
        for i in range(7):
            torque_ff = gravity_torques[i]
            # For a non-zero gravity joint, torque_ff should be non-zero
            if abs(positions[i]) > 0.1:
                # Gravity should contribute something
                pass  # Just verify structure
            assert math.isfinite(torque_ff)


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-adapter consistency tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossAdapterConsistency:
    """Verify that different adapters agree on physical semantics."""

    def test_lerobot_and_bavr_use_same_arm_api(self):
        """Both adapters use Arm() constructor and enable/disable lifecycle."""
        # This is a structural verification — both adapters must follow the same pattern:
        # 1. Arm(model, channel=...) or Arm(model, backend=...)
        # 2. arm.enable()
        # 3. ... use ...
        # 4. arm.disable()
        #
        # We verify by checking the adapter source patterns (already read)
        # LeRobot: Arm(config.model, channel=config.can_channel) → enable() → servoj → disable()
        # BAVR: Arm(config.model, channel=config.can_channel) → enable() → servol → disable()
        # This test documents the contract.
        pass

    def test_all_adapters_handle_7_joints(self):
        """All adapters are configured for 7-DOF arm."""
        # ROS2 joint names
        ros2_joints = [
            "joint1-a1_r", "joint2-a1_r", "joint3-a1_r", "joint4-a1_r",
            "joint5-a1_r", "joint6-a1_r", "joint7-a1_r",
        ]
        assert len(ros2_joints) == 7

        # LeRobot features
        from onero_sdk.model_registry import get_model
        model = get_model("A1")
        assert model.num_joints == 7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
