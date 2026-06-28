"""
Force mode algorithm tests — gravity compensation, task frame, selection.

Validates:
- Force mode applies force only in selected axes
- Non-selected axes hold position (stiffness)
- Gravity compensation uses correct joint configuration
- task_frame rotates the selection vector correctly
- Side-mount gravity compensation direction

Requires MuJoCo simulation: pip install onero-sdk[sim]

Boundary values:
- All axes selected (pure force)
- No axes selected (pure hold)
- Single axis at a time
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

try:
    import mujoco  # noqa: F401
    from onero_sdk import Arm
    from onero_sdk.mujoco_backend import MujocoBackend
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False

pytestmark = pytest.mark.skipif(not HAS_MUJOCO, reason="MuJoCo not installed")


@pytest.fixture
def arm():
    backend = MujocoBackend(model_name="A1")
    arm = Arm("A1", backend=backend)
    arm.enable()
    arm.movej([0.0, -0.5, 0.3, -1.0, 0.0, 0.5, 0.0], duration=2.0)
    yield arm
    arm.disable()
    backend.close()


@pytest.fixture
def arm_side():
    backend = MujocoBackend(model_name="A1")
    arm = Arm("A1", backend=backend, mounting="side_mount_left")
    arm.enable()
    arm.movej([0.0, -0.5, 0.3, -1.0, 0.0, 0.5, 0.0], duration=2.0)
    yield arm
    arm.disable()
    backend.close()


class TestForceModeBehavior:
    """Force mode moves arm in selected directions, holds in others."""

    def test_force_mode_z_force_moves_down(self, arm):
        """Applying -Z force moves arm downward."""
        pos_before, _ = arm.get_pose()

        for _ in range(200):
            arm.force_mode(selection=[0, 0, 1, 0, 0, 0], wrench=[0, 0, -5, 0, 0, 0])
        arm.force_mode_disable()

        pos_after, _ = arm.get_pose()
        dz = pos_after[2] - pos_before[2]

        assert dz < -0.005, (
            f"Z force should move arm down: dz={dz*1000:.1f}mm"
        )

    def test_force_mode_holds_unselected_axes(self, arm):
        """Non-selected axes maintain position."""
        pos_before, _ = arm.get_pose()

        # Only Z is force-controlled
        for _ in range(100):
            arm.force_mode(selection=[0, 0, 1, 0, 0, 0], wrench=[0, 0, -5, 0, 0, 0])
        arm.force_mode_disable()

        pos_after, _ = arm.get_pose()

        # X and Y should not drift much
        dx = abs(pos_after[0] - pos_before[0])
        dy = abs(pos_after[1] - pos_before[1])
        assert dx < 0.01, f"X drifted during Z force mode: {dx*1000:.1f}mm"
        assert dy < 0.01, f"Y drifted during Z force mode: {dy*1000:.1f}mm"

    def test_force_mode_no_selection_holds_all(self, arm):
        """selection=[0,0,0,0,0,0] with wrench: arm holds position."""
        pos_before, _ = arm.get_pose()

        for _ in range(100):
            arm.force_mode(selection=[0, 0, 0, 0, 0, 0], wrench=[0, 0, -5, 0, 0, 0])
        arm.force_mode_disable()

        pos_after, _ = arm.get_pose()
        drift = np.linalg.norm(pos_after - pos_before)
        assert drift < 0.005, (
            f"Arm drifted with no selection: {drift*1000:.1f}mm"
        )

    def test_force_mode_x_force_direction(self, arm):
        """+X force moves arm in +X direction."""
        pos_before, _ = arm.get_pose()

        for _ in range(200):
            arm.force_mode(selection=[1, 0, 0, 0, 0, 0], wrench=[5, 0, 0, 0, 0, 0])
        arm.force_mode_disable()

        pos_after, _ = arm.get_pose()
        dx = pos_after[0] - pos_before[0]
        assert dx > 0.005, f"+X force should move arm in +X: dx={dx*1000:.1f}mm"

    def test_force_mode_disable_stops_motion(self, arm):
        """After force_mode_disable, arm holds position."""
        # Apply force
        for _ in range(100):
            arm.force_mode(selection=[0, 0, 1, 0, 0, 0], wrench=[0, 0, -10, 0, 0, 0])
        arm.force_mode_disable()

        # Record position after disable
        pos_after_disable, _ = arm.get_pose()

        # Wait some frames — arm should hold
        time.sleep(0.1)
        pos_later, _ = arm.get_pose()
        drift = np.linalg.norm(pos_later - pos_after_disable)
        assert drift < 0.01, (
            f"Arm drifted after force_mode_disable: {drift*1000:.1f}mm"
        )


class TestGravityCompensation:
    """Gravity compensation uses current configuration, not stale state."""

    def test_gravity_different_for_different_configs(self, arm):
        """Gravity torques vary with configuration (not a constant)."""
        # Record torques at current config
        state1 = arm.state
        torques1 = list(state1.torques)

        # Move to different configuration
        arm.movej([0.5, -1.0, 0.5, -0.5, 0.3, 0.8, 0.0], duration=2.0)
        state2 = arm.state
        torques2 = list(state2.torques)

        # Torques should differ (gravity load changes with posture)
        diff = sum(abs(torques2[j] - torques1[j]) for j in range(7))
        assert diff > 0.1, (
            "Gravity torques identical in different configurations — "
            "possible stale state"
        )

    def test_side_mount_gravity_direction(self, arm_side):
        """Side-mounted arm: gravity in Y direction (not Z)."""
        gravity = arm_side._gravity_in_base
        # For side_mount_left, gravity should be primarily in Y
        assert abs(gravity[1]) > 5.0, (
            f"Side mount gravity Y component too small: {gravity}"
        )
        assert abs(gravity[2]) < 1.0, (
            f"Side mount gravity Z component should be near zero: {gravity}"
        )


class TestForceModeBoundary:
    """Boundary values for force mode."""

    def test_force_mode_zero_wrench(self, arm):
        """Zero wrench with selection: arm holds (no net force)."""
        pos_before, _ = arm.get_pose()

        for _ in range(100):
            arm.force_mode(selection=[1, 1, 1, 0, 0, 0], wrench=[0, 0, 0, 0, 0, 0])
        arm.force_mode_disable()

        pos_after, _ = arm.get_pose()
        # Even with all translational axes selected, zero force = hold
        # (gravity compensation should prevent drift)
        drift = np.linalg.norm(pos_after - pos_before)
        assert drift < 0.02, (
            f"Zero-wrench force mode drifted: {drift*1000:.1f}mm "
            "(gravity compensation may be incorrect)"
        )

    def test_force_mode_opposite_directions(self, arm):
        """Force in +Z then -Z: net displacement near zero."""
        pos_start, _ = arm.get_pose()

        # Push down
        for _ in range(100):
            arm.force_mode(selection=[0, 0, 1, 0, 0, 0], wrench=[0, 0, -5, 0, 0, 0])
        arm.force_mode_disable()

        # Push up
        for _ in range(100):
            arm.force_mode(selection=[0, 0, 1, 0, 0, 0], wrench=[0, 0, 5, 0, 0, 0])
        arm.force_mode_disable()

        pos_end, _ = arm.get_pose()
        dz = abs(pos_end[2] - pos_start[2])
        # Should approximately return (not exactly due to dynamics)
        assert dz < 0.03, (
            f"Opposite forces didn't approximately cancel: dz={dz*1000:.1f}mm"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
