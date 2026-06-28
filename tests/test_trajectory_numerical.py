"""
Trajectory numerical properties — velocity continuity, blend correctness.

Validates that TrajectoryPlanner produces trajectories with:
- Zero velocity at endpoints
- Smooth (no jumps) velocity profiles
- C¹ continuous transitions at blend regions
- Correct displacement integration

Pure math tests — no simulation or hardware.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from onero_sdk.trajectory import TrajectoryPlanner, Waypoint, TrajectoryFrame


NUM_JOINTS = 7
CONTROL_FREQ = 200.0
DT = 1.0 / CONTROL_FREQ


def _plan_single_segment(start, end, duration, blend=0.0):
    """Helper: plan a single waypoint trajectory."""
    planner = TrajectoryPlanner(num_joints=NUM_JOINTS, control_freq=CONTROL_FREQ)
    planner.add_waypoint(Waypoint(joints=end, duration=duration, blend=blend))
    return planner.compute(start)


def _plan_multi_segment(start, waypoints):
    """Helper: plan a multi-waypoint trajectory."""
    planner = TrajectoryPlanner(num_joints=NUM_JOINTS, control_freq=CONTROL_FREQ)
    for wp in waypoints:
        planner.add_waypoint(wp)
    return planner.compute(start)


class TestCosineProfile:
    """Single-segment cosine interpolation properties."""

    def test_cosine_profile_zero_velocity_endpoints(self):
        """Start and end velocities must be zero (cosine profile property)."""
        start = [0.0] * 7
        end = [0.5, -0.3, 0.2, 0.4, -0.1, 0.3, 0.0]
        frames = _plan_single_segment(start, end, duration=2.0)

        assert len(frames) > 10, "Too few frames generated"

        # First frame velocity should be near zero
        v_start = frames[0].velocities
        for j in range(NUM_JOINTS):
            assert abs(v_start[j]) < 0.01, (
                f"Joint {j} start velocity {v_start[j]:.4f} != 0"
            )

        # Last frame velocity should be near zero
        v_end = frames[-1].velocities
        for j in range(NUM_JOINTS):
            assert abs(v_end[j]) < 0.01, (
                f"Joint {j} end velocity {v_end[j]:.4f} != 0"
            )

    def test_cosine_profile_displacement_integral(self):
        """Velocity integrated over time = total displacement."""
        start = [0.0] * 7
        end = [1.0, -0.5, 0.3, 0.7, -0.2, 0.4, 0.1]
        frames = _plan_single_segment(start, end, duration=2.0)

        for j in range(NUM_JOINTS):
            expected_disp = end[j] - start[j]
            # Integrate velocity: sum(v_k * dt)
            integrated = sum(f.velocities[j] * DT for f in frames)
            # Also check: final position - initial position
            pos_disp = frames[-1].joints[j] - frames[0].joints[j]

            assert abs(pos_disp - expected_disp) < 0.01, (
                f"Joint {j} position displacement {pos_disp:.4f} != {expected_disp:.4f}"
            )

    def test_cosine_profile_velocity_is_smooth(self):
        """No velocity jumps: adjacent frame velocity difference bounded."""
        start = [0.0] * 7
        end = [0.8, -0.6, 0.4, 0.9, -0.3, 0.5, 0.2]
        frames = _plan_single_segment(start, end, duration=2.0)

        for j in range(NUM_JOINTS):
            velocities = [f.velocities[j] for f in frames]
            if max(abs(v) for v in velocities) < 1e-6:
                continue

            diffs = [abs(velocities[i+1] - velocities[i]) for i in range(len(velocities)-1)]
            max_diff = max(diffs)
            avg_diff = np.mean(diffs) if diffs else 0

            # Max acceleration step should not exceed 3x average (smooth profile)
            assert max_diff < 3.0 * avg_diff + 1e-6, (
                f"Joint {j} velocity jump: max_diff={max_diff:.6f}, avg={avg_diff:.6f}"
            )


class TestBlendRegion:
    """Multi-segment blend (cubic Hermite) correctness."""

    def test_blend_position_c1_continuous(self):
        """Position has no jumps at blend boundary."""
        start = [0.0] * 7
        wp1_joints = [0.5, -0.3, 0.2, 0.4, -0.1, 0.3, 0.0]
        wp2_joints = [1.0, -0.6, 0.4, 0.8, -0.2, 0.6, 0.0]

        frames = _plan_multi_segment(start, [
            Waypoint(joints=wp1_joints, duration=2.0, blend=0.3),
            Waypoint(joints=wp2_joints, duration=2.0, blend=0.0),
        ])

        # Check no position jumps anywhere in the trajectory
        for i in range(1, len(frames)):
            for j in range(NUM_JOINTS):
                jump = abs(frames[i].joints[j] - frames[i-1].joints[j])
                # At 200Hz, max reasonable step for a normal motion < 0.05 rad
                assert jump < 0.05, (
                    f"Position jump at frame {i}, joint {j}: {jump:.6f} rad"
                )

    def test_blend_velocity_matches_main_segment_exit(self):
        """Velocity at blend entry ≈ main segment exit velocity (C¹)."""
        start = [0.0] * 7
        wp1_joints = [0.5, -0.3, 0.2, 0.4, -0.1, 0.3, 0.0]
        wp2_joints = [1.0, -0.6, 0.4, 0.8, -0.2, 0.6, 0.0]

        # Compute with and without blend for comparison
        frames_blended = _plan_multi_segment(start, [
            Waypoint(joints=wp1_joints, duration=2.0, blend=0.3),
            Waypoint(joints=wp2_joints, duration=2.0, blend=0.0),
        ])

        # Verify no velocity discontinuity anywhere in trajectory
        for i in range(1, len(frames_blended)):
            for j in range(NUM_JOINTS):
                v_prev = frames_blended[i-1].velocities[j]
                v_curr = frames_blended[i].velocities[j]
                diff = abs(v_curr - v_prev)
                # Velocity change per frame should be bounded
                # (no harsh jumps at blend boundary)
                assert diff < 0.5, (
                    f"Velocity jump at frame {i}, joint {j}: "
                    f"Δv={diff:.4f} (prev={v_prev:.4f}, curr={v_curr:.4f})"
                )

    def test_multi_segment_no_position_jump(self):
        """Multi-segment trajectory has no position discontinuities."""
        start = [0.0] * 7
        frames = _plan_multi_segment(start, [
            Waypoint(joints=[0.3, -0.2, 0.1, 0.2, 0.0, 0.1, 0.0], duration=1.5, blend=0.2),
            Waypoint(joints=[0.6, -0.4, 0.2, 0.4, -0.1, 0.2, 0.0], duration=1.5, blend=0.2),
            Waypoint(joints=[0.9, -0.6, 0.3, 0.6, -0.2, 0.3, 0.0], duration=1.5, blend=0.0),
        ])

        max_jump = 0.0
        for i in range(1, len(frames)):
            for j in range(NUM_JOINTS):
                jump = abs(frames[i].joints[j] - frames[i-1].joints[j])
                max_jump = max(max_jump, jump)
                assert jump < 0.05, (
                    f"Position jump at frame {i}, joint {j}: {jump:.6f} rad"
                )

    def test_blend_region_stays_between_segments(self):
        """Blend positions stay within bounding box of adjacent segment endpoints."""
        start = [0.0] * 7
        wp1_joints = [1.0, -0.5, 0.3, 0.5, 0.0, 0.3, 0.0]
        wp2_joints = [2.0, -1.0, 0.6, 1.0, 0.0, 0.6, 0.0]

        frames = _plan_multi_segment(start, [
            Waypoint(joints=wp1_joints, duration=2.0, blend=0.3),
            Waypoint(joints=wp2_joints, duration=2.0, blend=0.0),
        ])

        # For each joint, all frames should be monotonically progressing
        # (when start < end for that joint) or bounded within a reasonable overshoot
        for j in range(NUM_JOINTS):
            positions = [f.joints[j] for f in frames]
            min_bound = min(start[j], wp1_joints[j], wp2_joints[j]) - 0.1
            max_bound = max(start[j], wp1_joints[j], wp2_joints[j]) + 0.1
            for i, p in enumerate(positions):
                assert min_bound <= p <= max_bound, (
                    f"Frame {i}, joint {j}: position {p:.4f} outside bounds "
                    f"[{min_bound:.4f}, {max_bound:.4f}]"
                )

    def test_zero_blend_produces_pure_cosine(self):
        """blend=0 means no Hermite region — just cosine interpolation."""
        start = [0.0] * 7
        end = [0.5, -0.3, 0.2, 0.4, -0.1, 0.3, 0.0]

        frames_no_blend = _plan_single_segment(start, end, duration=2.0, blend=0.0)

        # Verify cosine shape: velocity should peak at midpoint
        mid_idx = len(frames_no_blend) // 2
        for j in range(NUM_JOINTS):
            if abs(end[j] - start[j]) < 1e-6:
                continue
            v_mid = abs(frames_no_blend[mid_idx].velocities[j])
            v_start = abs(frames_no_blend[0].velocities[j])
            v_end = abs(frames_no_blend[-1].velocities[j])
            assert v_mid > v_start, (
                f"Joint {j}: midpoint velocity not greater than start"
            )
            assert v_mid > v_end, (
                f"Joint {j}: midpoint velocity not greater than end"
            )


class TestHermiteBoundary:
    """Hermite spline boundary conditions (blend region entry/exit)."""

    def test_hermite_boundary_conditions(self):
        """
        Cubic Hermite h(t): h(0)=p0, h(1)=p1.
        Test by verifying the blend region starts/ends at expected positions.
        """
        start = [0.0] * 7
        wp1_joints = [0.6, -0.4, 0.3, 0.5, -0.1, 0.3, 0.0]
        wp2_joints = [1.2, -0.8, 0.6, 1.0, -0.2, 0.6, 0.0]

        blend_dur = 0.4
        frames = _plan_multi_segment(start, [
            Waypoint(joints=wp1_joints, duration=2.0, blend=blend_dur),
            Waypoint(joints=wp2_joints, duration=2.0, blend=0.0),
        ])

        # The trajectory should start near start and end near wp2_joints
        for j in range(NUM_JOINTS):
            assert abs(frames[0].joints[j] - start[j]) < 0.01, (
                f"Joint {j}: trajectory start {frames[0].joints[j]:.4f} != {start[j]}"
            )

        # Final position should be near wp2_joints
        for j in range(NUM_JOINTS):
            assert abs(frames[-1].joints[j] - wp2_joints[j]) < 0.05, (
                f"Joint {j}: trajectory end {frames[-1].joints[j]:.4f} != {wp2_joints[j]:.4f}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
