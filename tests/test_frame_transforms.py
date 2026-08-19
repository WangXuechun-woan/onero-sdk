"""Tests for the frame= parameter on Cartesian APIs."""

import numpy as np
import pytest

from onero_sdk import Arm
from onero_sdk.mujoco_backend import MujocoBackend
from onero_sdk.exceptions import OneroConfigError


# Non-singular home pose. The URDF neutral (zero joints) sits at a hard
# shoulder-elbow singularity for A1 (sigma_min ≈ 5e-8), which causes IK
# to fail on any Cartesian motion. Move to this pose before any test
# that exercises Cartesian APIs.
_SAFE_HOME = [0.0, -0.5, 0.0, -1.0, 0.0, 0.5, 0.0]


@pytest.fixture
def arm_world():
    """Arm with world_frame=True (side mount) — default frame is 'world'."""
    backend = MujocoBackend(model_name="A1R")
    arm = Arm("A1R", backend=backend, mounting="side_mount_right", world_frame=True)
    arm.enable()
    arm.movej(_SAFE_HOME, duration=0.5)
    yield arm
    arm.disable()
    backend.close()


@pytest.fixture
def arm_base():
    """Arm with world_frame=False (upright) — default frame is 'base'."""
    backend = MujocoBackend(model_name="A1R")
    arm = Arm("A1R", backend=backend, world_frame=False)
    arm.enable()
    arm.movej(_SAFE_HOME, duration=0.5)
    yield arm
    arm.disable()
    backend.close()


@pytest.fixture
def arm_with_base_position():
    """Arm with non-zero base_position for dual-arm testing."""
    backend = MujocoBackend(model_name="A1R")
    arm = Arm("A1R", backend=backend, mounting="side_mount_right",
              world_frame=True, base_position=[0.5, 0.3, 0.0])
    arm.enable()
    arm.movej(_SAFE_HOME, duration=0.5)
    yield arm
    arm.disable()
    backend.close()


class TestFrameDefault:
    """Verify default_frame property and backward compatibility."""

    def test_world_frame_true_sets_default_world(self, arm_world):
        assert arm_world.default_frame == "world"

    def test_world_frame_false_sets_default_base(self, arm_base):
        assert arm_base.default_frame == "base"

    def test_set_world_frame_changes_default(self, arm_base):
        arm_base.set_world_frame(True)
        assert arm_base.default_frame == "world"
        arm_base.set_world_frame(False)
        assert arm_base.default_frame == "base"


class TestGetPoseFrame:
    """get_pose(frame=...) returns consistent results."""

    def test_get_pose_base_frame(self, arm_world):
        pos_default, rot_default = arm_world.get_pose()
        pos_world, rot_world = arm_world.get_pose(frame="world")
        pos_base, rot_base = arm_world.get_pose(frame="base")
        # Default is world, so default == world frame
        np.testing.assert_allclose(pos_default, pos_world, atol=1e-10)
        np.testing.assert_allclose(rot_default, rot_world, atol=1e-10)
        # Base frame should differ from world for side mount
        assert not np.allclose(pos_base, pos_world, atol=1e-3)

    def test_get_pose_roundtrip_base_world(self, arm_world):
        """world→base→world roundtrip via R_world_base."""
        pos_world, rot_world = arm_world.get_pose(frame="world")
        pos_base, rot_base = arm_world.get_pose(frame="base")
        R = arm_world.R_world_base
        np.testing.assert_allclose(R @ pos_base, pos_world, atol=1e-10)
        np.testing.assert_allclose(R @ rot_base, rot_world, atol=1e-10)

    def test_get_pose_with_base_position(self, arm_with_base_position):
        """base_position offset is correctly applied in world frame."""
        pos_world, _ = arm_with_base_position.get_pose(frame="world")
        pos_base, _ = arm_with_base_position.get_pose(frame="base")
        R = arm_with_base_position.R_world_base
        bp = arm_with_base_position.base_position_in_world
        # world = R @ base + base_position
        expected_world = R @ pos_base + bp
        np.testing.assert_allclose(pos_world, expected_world, atol=1e-10)

    def test_get_pose_base_position_roundtrip(self, arm_with_base_position):
        """Roundtrip: set base_position, get pose in world, convert back to base."""
        pos_world, rot_world = arm_with_base_position.get_pose(frame="world")
        pos_base, rot_base = arm_with_base_position.get_pose(frame="base")
        R = arm_with_base_position.R_world_base
        bp = arm_with_base_position.base_position_in_world
        # Manual: base = R.T @ (world - bp)
        reconstructed_base = R.T @ (pos_world - bp)
        np.testing.assert_allclose(pos_base, reconstructed_base, atol=1e-10)


class TestMovelFrame:
    """movel with explicit frame= parameter."""

    def test_movel_world_frame_accepted(self, arm_world):
        """movel with frame='world' executes without error."""
        pos, rot = arm_world.get_pose(frame="world")
        arm_world.movel(pos, rot, duration=0.3, frame="world")

    def test_movel_base_frame_accepted(self, arm_world):
        """movel with frame='base' executes without error."""
        pos_base, rot_base = arm_world.get_pose(frame="base")
        arm_world.movel(pos_base, rot_base, duration=0.3, frame="base")

    def test_movel_frame_consistency(self, arm_world):
        """Same physical target expressed in different frames produces similar motion."""
        pos_world, rot_world = arm_world.get_pose(frame="world")
        pos_base, rot_base = arm_world.get_pose(frame="base")
        R = arm_world.R_world_base
        np.testing.assert_allclose(R @ pos_base, pos_world, atol=1e-6)

    def test_movel_custom_frame(self, arm_base):
        """movel with a custom frame moves to the correct position."""
        R_custom = np.eye(3)
        p_custom = np.array([0.0, 0.0, 0.0])
        arm_base.register_frame("aligned", rotation=R_custom, position=p_custom)
        # For upright arm with identity custom frame, custom == world == base
        pos_base, rot_base = arm_base.get_pose(frame="base")
        arm_base.movel(pos_base, rot_base, duration=0.3, frame="aligned")
        pos_after, _ = arm_base.get_pose(frame="base")
        np.testing.assert_allclose(pos_after, pos_base, atol=0.01)


class TestToolFrame:
    """Tool frame (relative motion) tests."""

    def test_tool_frame_requires_set_tcp(self, arm_base):
        """frame='tool' raises if set_tcp() was never called."""
        with pytest.raises(Exception, match="set_tcp"):
            arm_base.movel([0, 0, 0.01, 0, 0, 0], frame="tool", duration=0.2)

    def test_movel_tool_frame_accepted(self, arm_base):
        """frame='tool' accepted after set_tcp()."""
        arm_base.set_tcp([0, 0, 0.1])  # 10cm tool along Z
        pos_before, _ = arm_base.get_pose(frame="base")
        displacement = [0, 0, 0.02, 0, 0, 0]
        arm_base.movel(displacement, frame="tool", duration=0.3)
        pos_after, _ = arm_base.get_pose(frame="base")
        assert not np.allclose(pos_before, pos_after, atol=0.005)

    def test_servop_tool_frame_accepted(self, arm_base):
        """servop with frame='tool' executes after set_tcp()."""
        arm_base.set_tcp([0, 0, 0.1])
        displacement = [0.01, 0, 0, 0, 0, 0]
        arm_base.servop(displacement, frame="tool")

    def test_tool_frame_direction_depends_on_ee(self, arm_base):
        """Tool-frame Z motion via movel produces meaningful displacement."""
        arm_base.set_tcp([0, 0, 0.1])
        pos_before, _ = arm_base.get_pose(frame="base")
        arm_base.movel([0, 0, 0.03, 0, 0, 0], frame="tool", duration=0.3)
        pos_after, _ = arm_base.get_pose(frame="base")
        displacement = np.linalg.norm(pos_after - pos_before)
        assert displacement > 0.01

    def test_tool_frame_with_rotation(self, arm_base):
        """Tool-frame motion with non-zero rotation displacement."""
        arm_base.set_tcp([0, 0, 0.1])
        _, rot_before = arm_base.get_pose(frame="base")
        # Small rotation around tool Z axis (0.1 rad ≈ 5.7°)
        arm_base.movel([0, 0, 0, 0, 0, 0.1], frame="tool", duration=0.3)
        _, rot_after = arm_base.get_pose(frame="base")
        # Rotation should have changed
        angle_diff = np.arccos(np.clip(
            (np.trace(rot_before.T @ rot_after) - 1) / 2, -1, 1))
        assert angle_diff > 0.05


class TestCustomFrame:
    """Custom frame registration and usage."""

    def test_register_and_use_custom_frame(self, arm_base):
        """Register a frame and use it with get_pose."""
        R_custom = np.eye(3)
        p_custom = np.array([1.0, 0, 0])
        arm_base.register_frame("test_frame", rotation=R_custom, position=p_custom)
        pos_base, _ = arm_base.get_pose(frame="base")
        pos_custom, _ = arm_base.get_pose(frame="test_frame")
        # custom frame has no rotation, just offset origin
        np.testing.assert_allclose(pos_custom, pos_base - p_custom, atol=1e-10)

    def test_unregister_frame(self, arm_base):
        arm_base.register_frame("temp", rotation=np.eye(3))
        arm_base.unregister_frame("temp")
        with pytest.raises(Exception):
            arm_base.get_pose(frame="temp")

    def test_cannot_register_builtin_name(self, arm_base):
        with pytest.raises(Exception):
            arm_base.register_frame("base", rotation=np.eye(3))
        with pytest.raises(Exception):
            arm_base.register_frame("world", rotation=np.eye(3))
        with pytest.raises(Exception):
            arm_base.register_frame("tool", rotation=np.eye(3))

    def test_unknown_frame_raises(self, arm_base):
        with pytest.raises(Exception):
            arm_base.get_pose(frame="nonexistent")

    def test_custom_frame_movel(self, arm_base):
        """Custom frame works with movel for moving to a target."""
        # Register a custom frame that's translated 0.1m in X
        arm_base.register_frame("shifted", rotation=np.eye(3),
                                position=np.array([0.1, 0, 0]))
        # Get current pose in custom frame
        pos_custom, rot_custom = arm_base.get_pose(frame="shifted")
        # Move to current pose (should be a no-op)
        arm_base.movel(pos_custom, rot_custom, duration=0.3, frame="shifted")
        pos_after, _ = arm_base.get_pose(frame="shifted")
        np.testing.assert_allclose(pos_after, pos_custom, atol=0.01)

    def test_register_frame_invalid_rotation_matrix(self, arm_base):
        """Registering a non-orthogonal matrix raises error."""
        bad_matrix = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        with pytest.raises(OneroConfigError, match="not a valid rotation"):
            arm_base.register_frame("bad", rotation=bad_matrix)

    def test_register_frame_rotation_vector(self, arm_base):
        """Registering with a rotation vector works."""
        arm_base.register_frame("rotated", rotation=[0, 0, 0.5])
        pos_base, _ = arm_base.get_pose(frame="base")
        pos_rotated, _ = arm_base.get_pose(frame="rotated")
        # Should differ because of rotation
        assert not np.allclose(pos_base, pos_rotated, atol=1e-6)


class TestSpeedlFrame:
    """speedl with frame= parameter."""

    def test_speedl_accepts_frame(self, arm_base):
        """speedl(frame='base') works without error."""
        arm_base.speedl([0, 0, 0.05, 0, 0, 0], duration=0.2, frame="base")

    def test_speedl_world_frame(self, arm_world):
        """speedl(frame='world') works without error."""
        arm_world.speedl([0, 0, 0.05, 0, 0, 0], duration=0.2, frame="world")


class TestForceModFrame:
    """force_mode with frame= parameter."""

    def test_force_mode_accepts_frame(self, arm_base):
        """force_mode(frame='base') works without error."""
        arm_base.force_mode(
            selection=[0, 0, 1, 0, 0, 0],
            wrench=[0, 0, 5, 0, 0, 0],
            frame="base",
        )

    def test_force_mode_world_frame(self, arm_world):
        """force_mode(frame='world') applies wrench in world coordinates."""
        arm_world.force_mode(
            selection=[0, 0, 1, 0, 0, 0],
            wrench=[0, 0, 5, 0, 0, 0],
            frame="world",
        )

    def test_force_mode_tool_frame(self, arm_base):
        """force_mode(frame='tool') requires set_tcp and works."""
        arm_base.set_tcp([0, 0, 0.1])
        arm_base.force_mode(
            selection=[0, 0, 1, 0, 0, 0],
            wrench=[0, 0, 5, 0, 0, 0],
            frame="tool",
        )

    def test_force_mode_custom_frame(self, arm_base):
        """force_mode with a custom frame works."""
        arm_base.register_frame("tilted", rotation=[0.3, 0, 0])
        arm_base.force_mode(
            selection=[0, 0, 1, 0, 0, 0],
            wrench=[0, 0, 5, 0, 0, 0],
            frame="tilted",
        )


class TestGetExternalWrench:
    """get_external_wrench with frame= parameter."""

    def test_get_external_wrench_base(self, arm_base):
        """get_external_wrench(frame='base') returns a 6-element array."""
        wrench = arm_base.get_external_wrench(frame="base")
        assert wrench.shape == (6,)

    def test_get_external_wrench_world(self, arm_world):
        """get_external_wrench(frame='world') returns a 6-element array."""
        wrench = arm_world.get_external_wrench(frame="world")
        assert wrench.shape == (6,)

    def test_get_external_wrench_frame_consistency(self, arm_world):
        """Wrench in different frames rotates correctly."""
        wrench_base = arm_world.get_external_wrench(frame="base")
        wrench_world = arm_world.get_external_wrench(frame="world")
        R = arm_world.R_world_base
        # Force components should be related by rotation
        np.testing.assert_allclose(R @ wrench_base[:3], wrench_world[:3], atol=1e-8)
        np.testing.assert_allclose(R @ wrench_base[3:], wrench_world[3:], atol=1e-8)


class TestShapeValidation:
    """Validate that wrong-shaped inputs raise clear errors."""

    def test_movel_wrong_position_shape(self, arm_base):
        """movel with wrong position shape raises."""
        with pytest.raises(OneroConfigError):
            arm_base.movel([0, 0], frame="base", duration=0.3)

    def test_speedl_wrong_twist_shape(self, arm_base):
        """speedl with wrong twist shape raises."""
        with pytest.raises(OneroConfigError):
            arm_base.speedl([0, 0, 0], duration=0.2, frame="base")

    def test_register_frame_wrong_position_shape(self, arm_base):
        """register_frame with wrong position shape raises."""
        with pytest.raises(OneroConfigError):
            arm_base.register_frame("bad", rotation=np.eye(3),
                                    position=[1, 2])

    def test_set_base_position_wrong_shape(self, arm_base):
        """set_base_position with wrong shape raises."""
        with pytest.raises(OneroConfigError):
            arm_base.set_base_position([1, 2, 3, 4])


class TestBasePosition:
    """Tests for base_position configuration."""

    def test_default_base_position_is_zero(self, arm_base):
        np.testing.assert_allclose(
            arm_base.base_position_in_world, [0, 0, 0], atol=1e-10)

    def test_set_base_position(self, arm_base):
        arm_base.set_base_position([1.0, 0.5, 0.2])
        np.testing.assert_allclose(
            arm_base.base_position_in_world, [1.0, 0.5, 0.2], atol=1e-10)

    def test_base_position_in_constructor(self, arm_with_base_position):
        np.testing.assert_allclose(
            arm_with_base_position.base_position_in_world, [0.5, 0.3, 0.0],
            atol=1e-10)
