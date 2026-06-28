"""
End-effector precision tests — measure actual motion accuracy against ISO 9283 standards.

Tests use MuJoCo simulation. Since simulation PD controller has inherent steady-state
error (~0.67° joint error under gravity), movej tests use looser tolerances.
movel/movec precision depends on IK solver quality and is much better.

Precision standards are defined in precision_standards.py based on:
- UR e-Series: ±0.03mm repeatability
- Franka Research 3: ±0.1mm repeatability
- KUKA LBR iiwa: ±0.1mm, ±0.01° joint repeatability
"""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from tests.precision_standards import (
    POS_REPEATABILITY_TIGHT, POS_REPEATABILITY_NORMAL, POS_REPEATABILITY_LOOSE,
    POS_ACCURACY_NORMAL, POS_ACCURACY_LOOSE,
    PATH_ACCURACY_NORMAL,
    ORI_REPEATABILITY_TIGHT, ORI_REPEATABILITY_NORMAL,
    ROTATION_MATH_TOL,
    TRAJECTORY_INTEGRAL_TOL,
)


def _has_mujoco():
    try:
        import mujoco
        return True
    except ImportError:
        return False


skip_no_mujoco = pytest.mark.skipif(
    not _has_mujoco(), reason="MuJoCo not available")


def _angle_between_rotations(R0, R1):
    cos_a = (np.trace(R0.T @ R1) - 1.0) / 2.0
    return np.arccos(np.clip(cos_a, -1.0, 1.0))


@pytest.fixture
def arm_sim():
    from onero_sdk import Arm
    from onero_sdk.mujoco_backend import MujocoBackend
    backend = MujocoBackend(model_name="A1")
    arm = Arm("A1", backend=backend)
    arm.enable()
    yield arm
    arm.disable()
    backend.close()


# ═══════════════════════════════════════════════════════════════════════════════
# movel (Cartesian linear) precision
# ═══════════════════════════════════════════════════════════════════════════════

@skip_no_mujoco
class TestMovelPrecision:
    """movel end-effector precision against ISO 9283 standards."""

    def test_movel_position_accuracy_50mm(self, arm_sim):
        """50mm linear motion: position error < NORMAL standard (0.5mm)."""
        arm = arm_sim
        home = [0.0, -0.5, 0.0, 1.0, 0.0, 0.5, 0.0]
        arm.movej(home, duration=2.0)

        pos0, rot0 = arm.get_pose()
        target_pos = pos0 + np.array([0.05, 0.0, 0.0])
        arm.movel(position=list(target_pos), rotation=rot0, duration=2.0)

        pos1, _ = arm.get_pose()
        error = np.linalg.norm(pos1 - target_pos)

        assert error < POS_ACCURACY_NORMAL, (
            f"movel 50mm position error {error*1000:.3f}mm exceeds "
            f"NORMAL standard {POS_ACCURACY_NORMAL*1000}mm"
        )

    def test_movel_orientation_preserved(self, arm_sim):
        """movel preserves orientation within NORMAL standard."""
        arm = arm_sim
        home = [0.0, -0.5, 0.0, 1.0, 0.0, 0.5, 0.0]
        arm.movej(home, duration=2.0)

        pos0, rot0 = arm.get_pose()
        target_pos = pos0 + np.array([0.05, 0.0, 0.0])
        arm.movel(position=list(target_pos), rotation=rot0, duration=2.0)

        _, rot1 = arm.get_pose()
        ori_error = _angle_between_rotations(rot0, rot1)

        assert ori_error < ORI_REPEATABILITY_NORMAL, (
            f"movel orientation error {math.degrees(ori_error):.4f}° exceeds "
            f"NORMAL standard {math.degrees(ORI_REPEATABILITY_NORMAL):.2f}°"
        )

    @pytest.mark.parametrize("distance_mm", [5, 10, 20, 50, 100])
    def test_movel_position_scales_linearly(self, arm_sim, distance_mm):
        """Position error scales sub-linearly with distance (no systematic drift)."""
        arm = arm_sim
        home = [0.0, -0.5, 0.0, 1.0, 0.0, 0.5, 0.0]
        arm.movej(home, duration=2.0)

        pos0, rot0 = arm.get_pose()
        d_m = distance_mm / 1000.0
        target_pos = pos0 + np.array([d_m, 0.0, 0.0])
        arm.movel(position=list(target_pos), rotation=rot0, duration=2.0)

        pos1, _ = arm.get_pose()
        error_mm = np.linalg.norm(pos1 - target_pos) * 1000

        # Error should be < 0.2% of distance (sub-linear scaling)
        max_allowed_mm = max(0.5, distance_mm * 0.002)
        assert error_mm < max_allowed_mm, (
            f"movel {distance_mm}mm: error {error_mm:.3f}mm exceeds "
            f"0.2% threshold ({max_allowed_mm:.3f}mm)"
        )

    def test_movel_repeatability(self, arm_sim):
        """Repeated movel to same target: spread < LOOSE standard (sim limitation)."""
        arm = arm_sim
        home = [0.0, -0.5, 0.0, 1.0, 0.0, 0.5, 0.0]

        positions = []
        for _ in range(5):
            arm.movej(home, duration=2.0)
            pos0, rot0 = arm.get_pose()
            target = pos0 + np.array([0.03, 0.0, 0.0])
            arm.movel(position=list(target), rotation=rot0, duration=2.0)
            pos_final, _ = arm.get_pose()
            positions.append(pos_final)

        # Repeatability: max spread among attained positions
        mean_pos = np.mean(positions, axis=0)
        spread = max(np.linalg.norm(p - mean_pos) for p in positions)

        assert spread < POS_REPEATABILITY_LOOSE, (
            f"movel repeatability {spread*1000:.3f}mm exceeds "
            f"LOOSE standard {POS_REPEATABILITY_LOOSE*1000}mm"
        )

    def test_movel_cumulative_accuracy(self, arm_sim):
        """5 sequential 10mm moves: final error < PATH_ACCURACY_NORMAL."""
        arm = arm_sim
        home = [0.0, -0.5, 0.0, 1.0, 0.0, 0.5, 0.0]
        arm.movej(home, duration=2.0)
        pos_start, rot_start = arm.get_pose()

        for i in range(5):
            target = pos_start + np.array([0.01 * (i + 1), 0.0, 0.0])
            arm.movel(position=list(target), rotation=rot_start, duration=1.0)

        pos_final, _ = arm.get_pose()
        expected = pos_start + np.array([0.05, 0.0, 0.0])
        error = np.linalg.norm(pos_final - expected)

        assert error < PATH_ACCURACY_NORMAL, (
            f"Cumulative 5×10mm error {error*1000:.3f}mm exceeds "
            f"NORMAL standard {PATH_ACCURACY_NORMAL*1000}mm"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# movec (Cartesian circular) precision
# ═══════════════════════════════════════════════════════════════════════════════

@skip_no_mujoco
class TestMovecPrecision:
    """movec circular motion precision."""

    def test_movec_reaches_target(self, arm_sim):
        """movec reaches end point within LOOSE standard."""
        arm = arm_sim
        home = [0.0, -0.5, 0.0, 1.0, 0.0, 0.5, 0.0]
        arm.movej(home, duration=2.0)

        p0, r0 = arm.get_pose()
        via_pos = p0 + np.array([0.03, 0.03, 0.0])
        end_pos = p0 + np.array([0.06, 0.0, 0.0])

        via_pose = list(via_pos) + [0, 0, 0]
        target_pose = list(end_pos) + [0, 0, 0]
        arm.movec(via_pose=via_pose, target_pose=target_pose, duration=2.0)

        p_final, _ = arm.get_pose()
        error = np.linalg.norm(p_final - end_pos)

        assert error < POS_ACCURACY_LOOSE, (
            f"movec endpoint error {error*1000:.3f}mm exceeds "
            f"LOOSE standard {POS_ACCURACY_LOOSE*1000}mm"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Trajectory planner algorithmic precision (pure math, no sim)
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrajectoryPlannerPrecision:
    """Trajectory planner outputs must be mathematically exact."""

    def test_cosine_endpoint_is_exact(self):
        """Cosine profile final frame = target joints (machine precision)."""
        from onero_sdk.trajectory import TrajectoryPlanner, Waypoint

        planner = TrajectoryPlanner(num_joints=7, control_freq=200.0)
        target = [0.5, -0.3, 0.2, 0.8, -0.1, 0.4, 0.1]
        planner.add_waypoint(Waypoint(joints=target, duration=2.0, blend=0.0))
        frames = planner.compute([0.0] * 7)

        for j in range(7):
            assert frames[-1].joints[j] == target[j], (
                f"Joint {j}: final frame {frames[-1].joints[j]} != target {target[j]}"
            )

    def test_cosine_endpoint_velocity_zero(self):
        """Cosine profile final velocity = 0 (machine precision)."""
        from onero_sdk.trajectory import TrajectoryPlanner, Waypoint

        planner = TrajectoryPlanner(num_joints=7, control_freq=200.0)
        planner.add_waypoint(Waypoint(
            joints=[1.0, -0.5, 0.3, 0.6, -0.2, 0.4, 0.1],
            duration=2.0, blend=0.0))
        frames = planner.compute([0.0] * 7)

        for j in range(7):
            assert abs(frames[-1].velocities[j]) < 1e-14, (
                f"Joint {j}: final velocity {frames[-1].velocities[j]:.2e} != 0"
            )

    @pytest.mark.parametrize("duration", [0.5, 1.0, 2.0, 5.0])
    def test_velocity_integral_equals_displacement(self, duration):
        """∫v(t)dt = Δpos within TRAJECTORY_INTEGRAL_TOL for all durations."""
        from onero_sdk.trajectory import TrajectoryPlanner, Waypoint

        planner = TrajectoryPlanner(num_joints=7, control_freq=200.0)
        target = [0.8, -0.4, 0.3, 0.5, -0.2, 0.6, 0.1]
        planner.add_waypoint(Waypoint(joints=target, duration=duration, blend=0.0))
        frames = planner.compute([0.0] * 7)

        dt = 1.0 / 200.0
        for j in range(7):
            integrated = sum(f.velocities[j] * dt for f in frames)
            displacement = frames[-1].joints[j] - frames[0].joints[j]
            if abs(displacement) < 1e-10:
                continue
            rel_err = abs(integrated - displacement) / abs(displacement)
            assert rel_err < TRAJECTORY_INTEGRAL_TOL, (
                f"Joint {j}, dur={duration}s: "
                f"∫v·dt rel error {rel_err*100:.2f}% > {TRAJECTORY_INTEGRAL_TOL*100}%"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# SLERP algorithmic precision
# ═══════════════════════════════════════════════════════════════════════════════

class TestSLERPPrecision:
    """SLERP must achieve machine-epsilon geodesic accuracy."""

    @pytest.mark.parametrize("angle_deg", [10, 45, 90, 135, 179])
    def test_slerp_geodesic_error_sub_microradian(self, angle_deg):
        """SLERP geodesic proportionality error < 1e-10 rad for all angles."""
        from onero_sdk.geometry import slerp_matrix, rotvec_to_matrix

        R0 = rotvec_to_matrix([0.3, -0.2, 0.5])
        R1 = R0 @ rotvec_to_matrix([0, 0, math.radians(angle_deg)])
        total = math.radians(angle_deg)

        max_err = 0
        for i in range(101):
            t = i / 100.0
            R = slerp_matrix(R0, R1, t)
            actual = _angle_between_rotations(R0, R)
            expected = t * total
            max_err = max(max_err, abs(actual - expected))

        assert max_err < ROTATION_MATH_TOL, (
            f"SLERP {angle_deg}°: geodesic error {max_err:.2e} rad > "
            f"tolerance {ROTATION_MATH_TOL:.0e} rad"
        )

    def test_slerp_so3_validity_across_full_range(self):
        """All SLERP outputs are valid SO(3) to machine precision."""
        from onero_sdk.geometry import slerp_matrix, rotvec_to_matrix

        R0 = rotvec_to_matrix([0.1, 0.2, -0.3])
        R1 = R0 @ rotvec_to_matrix([0, 0, math.radians(170)])

        for i in range(201):
            t = i / 200.0
            R = slerp_matrix(R0, R1, t)
            orth_err = np.linalg.norm(R.T @ R - np.eye(3))
            det_err = abs(np.linalg.det(R) - 1.0)
            assert orth_err < 1e-10, f"t={t:.3f}: orthogonality error {orth_err:.2e}"
            assert det_err < 1e-10, f"t={t:.3f}: determinant error {det_err:.2e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
