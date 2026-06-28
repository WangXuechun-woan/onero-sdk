"""
IK solver algorithm tests — convergence, warm-start, recovery, boundaries.

Validates the DLS IK solver against known FK→IK round-trip invariants,
warm-start continuity, and behavior at joint limits and near singularities.

Requires pinocchio: pip install pin
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

try:
    import pinocchio as pin
except ImportError:
    pin = None

pytestmark = pytest.mark.skipif(pin is None, reason="pinocchio not installed")


def _get_urdf_path():
    """Resolve URDF path for A1 right arm."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "onero_sdk", "urdf", "A1", "a1_r.urdf")


@pytest.fixture
def ik_solver():
    from onero_sdk.ik_solver import IKSolver
    return IKSolver(urdf_path=_get_urdf_path(), ee_frame="Link7")


class TestIKConvergence:
    """IK solver converges to correct solutions."""

    def test_ik_converges_to_known_pose(self, ik_solver):
        """FK(solve(target)) ≈ target — round-trip consistency."""
        # Use a known reachable configuration
        q_known = np.array([0.3, -0.5, 0.2, -0.8, 0.1, 0.4, -0.1])
        pos_target, rot_target = ik_solver.get_ee_pose(q_known)

        # Solve IK from a different initial guess
        q_init = np.zeros(7)
        q_sol = ik_solver.solve_ik(
            target_pos=pos_target,
            target_rot=rot_target,
            q_init=q_init,
        )

        # Verify FK of solution matches target
        pos_sol, rot_sol = ik_solver.get_ee_pose(q_sol)
        pos_err = np.linalg.norm(pos_sol - pos_target)
        rot_err = np.linalg.norm(rot_sol - rot_target)

        assert pos_err < 1e-3, f"IK position error too large: {pos_err:.6f}m"
        assert rot_err < 1e-2, f"IK rotation error too large: {rot_err:.6f}rad"

    def test_ik_position_only_converges(self, ik_solver):
        """IK with only position target (no rotation) converges."""
        q_known = np.array([0.2, -0.4, 0.1, -0.6, 0.0, 0.3, 0.0])
        pos_target, _ = ik_solver.get_ee_pose(q_known)

        q_sol = ik_solver.solve_ik(
            target_pos=pos_target,
            target_rot=None,
            q_init=np.zeros(7),
        )

        pos_sol, _ = ik_solver.get_ee_pose(q_sol)
        pos_err = np.linalg.norm(pos_sol - pos_target)
        assert pos_err < 1e-3, f"Position-only IK error: {pos_err:.6f}m"


class TestIKWarmStart:
    """Warm-start (continuity) — consecutive targets produce continuous solutions."""

    def test_ik_warm_start_continuity(self, ik_solver):
        """
        Sequential nearby targets: solution joints change smoothly.
        Bug regression: without warm-start, IK can jump to distant solutions.
        """
        # Trace a short straight line in Cartesian space
        q_start = np.array([0.3, -0.5, 0.2, -0.8, 0.1, 0.4, -0.1])
        pos_start, rot_start = ik_solver.get_ee_pose(q_start)

        direction = np.array([0.01, 0.0, 0.0])  # 1cm steps in X
        prev_q = q_start.copy()
        ik_solver.q = q_start  # Set warm-start baseline

        for step in range(1, 10):
            target_pos = pos_start + step * direction
            q_sol = ik_solver.solve_ik(
                target_pos=target_pos,
                target_rot=rot_start,
            )
            # Joint change between consecutive solutions should be small
            dq = np.linalg.norm(q_sol - prev_q)
            assert dq < 0.2, (
                f"Step {step}: joint jump {dq:.4f} rad — warm-start not working"
            )
            prev_q = q_sol.copy()

    def test_ik_reset_recovers_from_failure(self, ik_solver):
        """After a failed solve, reset and re-solve from good initial guess works."""
        # First, move to a known good configuration
        q_good = np.array([0.2, -0.4, 0.1, -0.6, 0.0, 0.3, 0.0])
        pos_good, rot_good = ik_solver.get_ee_pose(q_good)

        # Try to solve for an unreachable target (very far away)
        pos_far = np.array([10.0, 10.0, 10.0])
        q_failed = ik_solver.solve_ik(
            target_pos=pos_far,
            target_rot=None,
            q_init=q_good,
            max_iter=5,  # few iterations → won't converge
        )

        # Now reset and solve for the original reachable target
        ik_solver.q = q_good  # Reset warm-start
        q_recovered = ik_solver.solve_ik(
            target_pos=pos_good,
            target_rot=rot_good,
            q_init=q_good,
        )

        pos_rec, _ = ik_solver.get_ee_pose(q_recovered)
        pos_err = np.linalg.norm(pos_rec - pos_good)
        assert pos_err < 1e-3, (
            f"Recovery after failure: pos_err={pos_err:.6f}m (should be < 1mm)"
        )


class TestIKJointLimits:
    """IK respects joint limits."""

    def test_ik_respects_joint_limits(self, ik_solver):
        """Solution joints always within model's position limits."""
        q_min, q_max = ik_solver.joint_limits

        # Solve for multiple targets
        test_configs = [
            np.array([0.3, -0.5, 0.2, -0.8, 0.1, 0.4, -0.1]),
            np.array([-0.3, 0.5, -0.2, 0.8, -0.1, -0.4, 0.1]),
            np.array([0.0, -1.0, 0.5, -1.5, 0.0, 0.8, 0.0]),
        ]

        for q_known in test_configs:
            pos, rot = ik_solver.get_ee_pose(q_known)
            q_sol = ik_solver.solve_ik(
                target_pos=pos,
                target_rot=rot,
                q_init=np.zeros(7),
            )

            for j in range(7):
                assert q_sol[j] >= q_min[j] - 1e-6, (
                    f"Joint {j} below lower limit: {q_sol[j]:.4f} < {q_min[j]:.4f}"
                )
                assert q_sol[j] <= q_max[j] + 1e-6, (
                    f"Joint {j} above upper limit: {q_sol[j]:.4f} > {q_max[j]:.4f}"
                )


class TestIKSingularity:
    """IK behavior near singular configurations."""

    def test_ik_near_singularity_finite(self, ik_solver):
        """Near singularity: DLS prevents infinite joint velocities."""
        # Fully extended arm — close to kinematic singularity
        q_sing = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        pos, rot = ik_solver.get_ee_pose(q_sing)

        # Slightly offset target (should still converge with DLS damping)
        pos_offset = pos + np.array([0.01, 0.01, 0.0])
        q_sol = ik_solver.solve_ik(
            target_pos=pos_offset,
            target_rot=None,
            q_init=q_sing,
        )

        # Solution should be finite and not blow up
        assert np.all(np.isfinite(q_sol)), "IK produced non-finite values near singularity"
        # Joint change should be bounded (DLS caps it)
        dq = np.linalg.norm(q_sol - q_sing)
        assert dq < 5.0, f"Excessive joint change near singularity: {dq:.4f}"


class TestIKConsistency:
    """IK solver produces consistent results."""

    def test_ik_consistent_for_same_target(self, ik_solver):
        """Same target + same initial guess → same solution (deterministic)."""
        q_known = np.array([0.3, -0.5, 0.2, -0.8, 0.1, 0.4, -0.1])
        pos, rot = ik_solver.get_ee_pose(q_known)
        q_init = np.zeros(7)

        q_sol1 = ik_solver.solve_ik(target_pos=pos, target_rot=rot, q_init=q_init)
        q_sol2 = ik_solver.solve_ik(target_pos=pos, target_rot=rot, q_init=q_init)

        assert np.allclose(q_sol1, q_sol2, atol=1e-10), (
            "Same inputs produced different IK solutions (non-deterministic)"
        )

    def test_ik_redundancy_different_nullspace(self, ik_solver):
        """Different initial guesses → different solutions but same EE pose (redundancy)."""
        # Position-only target: 7-DOF arm has infinite solutions
        q_ref = np.array([0.3, -0.5, 0.2, -0.8, 0.1, 0.4, -0.1])
        pos_target, _ = ik_solver.get_ee_pose(q_ref)

        # Two different initial guesses
        q_init1 = np.array([0.0, -0.3, 0.0, -0.5, 0.0, 0.2, 0.0])
        q_init2 = np.array([0.5, -0.8, 0.4, -1.2, 0.2, 0.6, -0.2])

        q_sol1 = ik_solver.solve_ik(target_pos=pos_target, target_rot=None, q_init=q_init1)
        q_sol2 = ik_solver.solve_ik(target_pos=pos_target, target_rot=None, q_init=q_init2)

        # Both should reach the target position
        pos1, _ = ik_solver.get_ee_pose(q_sol1)
        pos2, _ = ik_solver.get_ee_pose(q_sol2)
        assert np.linalg.norm(pos1 - pos_target) < 1e-3
        assert np.linalg.norm(pos2 - pos_target) < 1e-3

        # But joint configurations should differ (redundancy exploited)
        # (only check if initial guesses are sufficiently different)
        if np.linalg.norm(q_init1 - q_init2) > 0.5:
            diff = np.linalg.norm(q_sol1 - q_sol2)
            assert diff > 0.01, (
                "Different initial guesses converged to identical solution — "
                "redundancy not being exploited"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
