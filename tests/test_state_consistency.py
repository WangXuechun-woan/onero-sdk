"""
State consistency tests — sequential operations, mode transitions, recovery.

Validates:
- Motion after failed/interrupted motion still converges
- Mode transitions (force_mode → movel, speedl → movej) are clean
- IK warm-start remains consistent across operations
- Emergency stop / re-enable cycle works

These are integration tests that exercise the state machine transitions
that typically expose stale-state bugs.

Requires MuJoCo simulation: pip install onero-sdk[sim]
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
    backend = MujocoBackend(model_name="A1R")
    arm = Arm("A1R", backend=backend)
    arm.enable()
    yield arm
    arm.disable()
    backend.close()


class TestSequentialMotion:
    """Sequential motion operations maintain state consistency."""

    def test_movej_after_movej(self, arm):
        """Two consecutive movej: second reaches its target."""
        target1 = [0.3, -0.5, 0.2, -0.8, 0.1, 0.4, -0.1]
        target2 = [-0.2, -0.3, 0.1, -0.5, -0.1, 0.2, 0.1]

        arm.movej(target1, duration=2.0)
        arm.movej(target2, duration=2.0)

        errors = [abs(arm.state.joints[j] - target2[j]) for j in range(7)]
        max_err = max(errors)
        assert max_err < 0.05, (
            f"Second movej failed to converge: max_err={max_err:.4f} rad"
        )

    def test_movel_after_movej(self, arm):
        """movel after movej: Cartesian motion still works."""
        arm.movej([0.0, -0.5, 0.3, -1.0, 0.0, 0.5, 0.0], duration=2.0)

        pos_before, _ = arm.get_pose()
        target_pos = pos_before + np.array([0.03, 0.0, 0.0])
        target_6d = list(target_pos) + [0.0, 0.0, 0.0]

        arm.movel(position=target_6d, duration=2.0)
        pos_after, _ = arm.get_pose()

        err = np.linalg.norm(pos_after - target_pos)
        assert err < 0.01, f"movel after movej error: {err*1000:.1f}mm"

    def test_sequential_movel_ik_continuity(self, arm):
        """Two consecutive movels: IK warm-start keeps solutions consistent."""
        arm.movej([0.0, -0.5, 0.3, -1.0, 0.0, 0.5, 0.0], duration=2.0)
        pos, _ = arm.get_pose()

        # First movel
        t1 = list(pos + np.array([0.02, 0.0, 0.0])) + [0.0, 0.0, 0.0]
        arm.movel(position=t1, duration=1.5)
        joints_mid = list(arm.state.joints)

        # Second movel (continuing in same direction)
        t2 = list(pos + np.array([0.04, 0.0, 0.0])) + [0.0, 0.0, 0.0]
        arm.movel(position=t2, duration=1.5)
        joints_end = list(arm.state.joints)

        # Joints should be similar direction of change (no configuration flip)
        for j in range(7):
            jump = abs(joints_end[j] - joints_mid[j])
            assert jump < 1.0, (
                f"Joint {j} jumped {jump:.3f} rad between consecutive movels"
            )


class TestModeTransitions:
    """Mode transitions (force → position, velocity → position)."""

    def test_force_mode_then_movej(self, arm):
        """Force mode → movej: position control works after force mode."""
        arm.movej([0.0, -0.5, 0.3, -1.0, 0.0, 0.5, 0.0], duration=2.0)

        # Force mode phase
        for _ in range(100):
            arm.force_mode(selection=[0, 0, 1, 0, 0, 0], wrench=[0, 0, -3, 0, 0, 0])
        arm.force_mode_disable()

        # Now do movej — should work cleanly
        target = [0.2, -0.3, 0.1, -0.7, 0.0, 0.3, 0.0]
        arm.movej(target, duration=2.0)

        errors = [abs(arm.state.joints[j] - target[j]) for j in range(7)]
        max_err = max(errors)
        assert max_err < 0.05, (
            f"movej after force_mode failed: max_err={max_err:.4f} rad"
        )

    def test_force_mode_then_movel(self, arm):
        """Force mode → movel: Cartesian control works after force mode."""
        arm.movej([0.0, -0.5, 0.3, -1.0, 0.0, 0.5, 0.0], duration=2.0)

        # Force mode
        for _ in range(100):
            arm.force_mode(selection=[0, 0, 1, 0, 0, 0], wrench=[0, 0, -3, 0, 0, 0])
        arm.force_mode_disable()

        # movel
        pos, _ = arm.get_pose()
        target = list(pos + np.array([0.03, 0.0, 0.0])) + [0.0, 0.0, 0.0]
        arm.movel(position=target, duration=2.0)
        pos_end, _ = arm.get_pose()

        err = np.linalg.norm(pos_end - (pos + np.array([0.03, 0.0, 0.0])))
        assert err < 0.01, f"movel after force_mode error: {err*1000:.1f}mm"

    def test_speedl_then_movej(self, arm):
        """speedl → movej: position control works after velocity mode."""
        arm.movej([0.0, -0.5, 0.3, -1.0, 0.0, 0.5, 0.0], duration=2.0)

        # Velocity mode
        arm.speedl(velocities=[0.02, 0, 0, 0, 0, 0], duration=0.3)

        # Back to position mode
        target = [0.1, -0.4, 0.2, -0.9, 0.0, 0.4, 0.0]
        arm.movej(target, duration=2.0)

        errors = [abs(arm.state.joints[j] - target[j]) for j in range(7)]
        max_err = max(errors)
        assert max_err < 0.05, (
            f"movej after speedl failed: max_err={max_err:.4f} rad"
        )


class TestRecovery:
    """Recovery from abnormal states."""

    def test_home_after_various_motions(self, arm):
        """home() works regardless of previous motion history."""
        # Do various motions
        arm.movej([0.5, -0.8, 0.4, -1.2, 0.2, 0.6, -0.2], duration=1.5)
        arm.movej([0.0, -0.5, 0.3, -1.0, 0.0, 0.5, 0.0], duration=1.5)

        pos, _ = arm.get_pose()
        target_6d = list(pos + np.array([0.02, 0.0, 0.0])) + [0.0, 0.0, 0.0]
        arm.movel(position=target_6d, duration=1.5)

        # Home should bring arm to zero
        arm.home(duration=3.0)
        max_err = max(abs(j) for j in arm.state.joints)
        assert max_err < 0.05, (
            f"home() failed after motion sequence: max_err={max_err:.4f} rad"
        )

    def test_disable_reenable_motion(self, arm):
        """Disable → re-enable → motion still works."""
        arm.movej([0.3, -0.5, 0.2, -0.8, 0.1, 0.4, -0.1], duration=2.0)

        # Disable and re-enable
        arm.disable()
        arm.enable()

        # Motion should still work
        target = [0.0, -0.3, 0.1, -0.6, 0.0, 0.3, 0.0]
        arm.movej(target, duration=2.0)

        errors = [abs(arm.state.joints[j] - target[j]) for j in range(7)]
        max_err = max(errors)
        assert max_err < 0.05, (
            f"movej after disable/enable failed: max_err={max_err:.4f} rad"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
