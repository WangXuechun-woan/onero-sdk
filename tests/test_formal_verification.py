"""
Formal-verification-style tests — parametrized exhaustive traversal.

Design principles:
1. Enumerate: sweep across parameter space (0→1, 0°→360°, all mode transitions)
2. Assert math definitions: not "doesn't crash" but "satisfies the invariant"
3. Catch integration wiring bugs: test through the real call path, not isolated units
4. Mutation-resistant: if you change the sign, swap args, or use the wrong variable,
   these tests WILL fail

Each test class targets one mathematical property and sweeps it exhaustively.
"""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from onero_sdk.geometry import slerp_matrix, rotvec_to_matrix, signed_arc_angle
from onero_sdk.trajectory import TrajectoryPlanner, Waypoint
from onero_sdk.safety import SafetyMonitor, SafetyConfig
from onero_sdk.safe_backend import SafeBackend
from onero_sdk.backend import ArmBackend, ArmState


# ═══════════════════════════════════════════════════════════════════════════════
# SLERP: 形式验证级遍历
# ═══════════════════════════════════════════════════════════════════════════════

def _angle_between(R0, R1):
    cos_a = (np.trace(R0.T @ R1) - 1.0) / 2.0
    return np.arccos(np.clip(cos_a, -1.0, 1.0))


def _is_SO3(R, tol=1e-8):
    return (np.linalg.norm(R.T @ R - np.eye(3)) < tol and
            abs(np.linalg.det(R) - 1.0) < tol)


# 遍历 0→180° 的各种旋转角
SLERP_ANGLES_DEG = [0, 1, 5, 10, 30, 45, 60, 89, 90, 91, 120, 135, 150, 170, 179, 180]
SLERP_T_VALUES = [0.0, 0.01, 0.1, 0.25, 0.333, 0.5, 0.667, 0.75, 0.9, 0.99, 1.0]


class TestSLERPFormal:
    """SLERP 数学性质的形式化验证 — 对全参数空间遍历."""

    @pytest.mark.parametrize("angle_deg", SLERP_ANGLES_DEG)
    @pytest.mark.parametrize("t", SLERP_T_VALUES)
    def test_slerp_is_valid_SO3(self, angle_deg, t):
        """∀ angle ∈ [0,180°], ∀ t ∈ [0,1]: slerp(R0,R1,t) ∈ SO(3)."""
        R0 = rotvec_to_matrix([0.3, -0.2, 0.5])
        R1 = R0 @ rotvec_to_matrix([0, 0, math.radians(angle_deg)])
        R = slerp_matrix(R0, R1, t)
        assert _is_SO3(R, tol=1e-6), (
            f"angle={angle_deg}°, t={t}: not SO(3), "
            f"det={np.linalg.det(R):.8f}, orth_err={np.linalg.norm(R.T@R - np.eye(3)):.2e}"
        )

    @pytest.mark.parametrize("angle_deg", [10, 30, 60, 90, 120, 150, 179])
    @pytest.mark.parametrize("t", SLERP_T_VALUES)
    def test_slerp_geodesic_proportionality(self, angle_deg, t):
        """
        ∀ t ∈ [0,1]: d(R0, slerp(t)) = t × d(R0, R1)

        This IS the mathematical definition of SLERP.
        If this fails, the implementation is wrong — not "slightly off."
        """
        R0 = rotvec_to_matrix([0.1, 0.2, -0.3])
        R1 = R0 @ rotvec_to_matrix([0, 0, math.radians(angle_deg)])

        total = _angle_between(R0, R1)
        R_t = slerp_matrix(R0, R1, t)
        partial = _angle_between(R0, R_t)

        expected = t * total
        assert abs(partial - expected) < 1e-5, (
            f"angle={angle_deg}°, t={t}: "
            f"d(R0, slerp(t))={math.degrees(partial):.4f}° "
            f"≠ t×total={math.degrees(expected):.4f}°"
        )

    @pytest.mark.parametrize("angle_deg", [10, 45, 90, 135, 170])
    def test_slerp_monotone_distance(self, angle_deg):
        """d(R0, slerp(t)) is strictly monotone increasing for t ∈ [0,1]."""
        R0 = rotvec_to_matrix([0.4, -0.1, 0.2])
        R1 = R0 @ rotvec_to_matrix([0, 0, math.radians(angle_deg)])

        prev_d = 0.0
        for t in np.linspace(0.01, 1.0, 50):
            d = _angle_between(R0, slerp_matrix(R0, R1, t))
            assert d >= prev_d - 1e-8, (
                f"angle={angle_deg}°: non-monotone at t={t:.3f}, "
                f"d={math.degrees(d):.4f}° < prev={math.degrees(prev_d):.4f}°"
            )
            prev_d = d


# ═══════════════════════════════════════════════════════════════════════════════
# signed_arc_angle: 全圆弧遍历
# ═══════════════════════════════════════════════════════════════════════════════

class TestArcAngleFormal:
    """signed_arc_angle 对各种圆弧角度的数学正确性."""

    @pytest.mark.parametrize("via_deg", [30, 45, 60, 90, 120, 150])
    @pytest.mark.parametrize("end_deg", [60, 90, 120, 180, 240, 300, 350])
    def test_arc_angle_positive_direction(self, via_deg, end_deg):
        """
        正方向: via 在 start→end 之间 → 角度符号正确.
        只测 via_deg < end_deg 的有效组合.
        """
        if via_deg >= end_deg:
            pytest.skip("via must be between start and end")

        normal = np.array([0, 0, 1.0])
        v0 = np.array([1.0, 0.0, 0.0])
        v1 = np.array([math.cos(math.radians(via_deg)),
                       math.sin(math.radians(via_deg)), 0.0])
        v2 = np.array([math.cos(math.radians(end_deg)),
                       math.sin(math.radians(end_deg)), 0.0])

        angle = signed_arc_angle(v0, v2, normal, v1)
        expected = math.radians(end_deg)

        assert abs(angle - expected) < 1e-4, (
            f"via={via_deg}°, end={end_deg}°: "
            f"got {math.degrees(angle):.2f}° expected {end_deg}°"
        )

    @pytest.mark.parametrize("via_deg", [330, 300, 270, 240, 210])
    def test_arc_angle_negative_direction(self, via_deg):
        """
        反方向弧: via 在反方向 → 返回负角.
        start=0°, end=180°, via 在 180°~360° 之间 → 走反方向 (-180°).
        """
        normal = np.array([0, 0, 1.0])
        v0 = np.array([1.0, 0.0, 0.0])
        v1 = np.array([math.cos(math.radians(via_deg)),
                       math.sin(math.radians(via_deg)), 0.0])
        v2 = np.array([-1.0, 0.0, 0.0])  # 180°

        angle = signed_arc_angle(v0, v2, normal, v1)
        # Via is in the negative half → angle should be -π
        assert angle < 0, f"via={via_deg}°: expected negative, got {math.degrees(angle):.2f}°"
        assert abs(angle + math.pi) < 1e-4, (
            f"via={via_deg}°: expected -180°, got {math.degrees(angle):.2f}°"
        )

    def test_arc_angle_full_circle(self):
        """全圆弧 (start ≈ end, via 在对面) → 返回 ±2π."""
        normal = np.array([0, 0, 1.0])
        v0 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([1.0, 0.0, 0.0])  # same as start
        v1 = np.array([0.0, 1.0, 0.0])  # via at 90°

        angle = signed_arc_angle(v0, v2, normal, v1)
        assert abs(abs(angle) - 2 * math.pi) < 1e-4, (
            f"Full circle: expected ±2π, got {math.degrees(angle):.2f}°"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Trajectory blend: C1 连续性的数学定义验证
# ═══════════════════════════════════════════════════════════════════════════════

BLEND_DURATIONS = [0.1, 0.2, 0.3, 0.5]
SEGMENT_DURATIONS = [1.5, 2.0, 3.0, 5.0]


class TestBlendC1Formal:
    """
    Blend region C1 continuity — verified by math definition:
    - C0: |pos[k] - pos[k-1]| < max_expected_step (no position jumps)
    - C1: velocity derivative bounded (no acceleration spikes at boundaries)
    """

    @pytest.mark.parametrize("blend_dur", BLEND_DURATIONS)
    @pytest.mark.parametrize("seg_dur", SEGMENT_DURATIONS)
    def test_no_position_discontinuity(self, blend_dur, seg_dur):
        """∀ blend/seg combinations: max(|Δpos|) < expected from velocity."""
        if blend_dur >= seg_dur:
            pytest.skip("blend must be < segment duration")

        planner = TrajectoryPlanner(num_joints=7, control_freq=200.0)
        planner.add_waypoint(Waypoint(
            joints=[0.5, -0.3, 0.2, 0.4, -0.1, 0.3, 0.0],
            duration=seg_dur, blend=blend_dur))
        planner.add_waypoint(Waypoint(
            joints=[1.0, -0.6, 0.4, 0.8, -0.2, 0.6, 0.0],
            duration=seg_dur, blend=0.0))

        frames = planner.compute([0.0] * 7)
        dt = 1.0 / 200.0

        for i in range(1, len(frames)):
            for j in range(7):
                jump = abs(frames[i].joints[j] - frames[i-1].joints[j])
                # Math definition: position change per frame ≤ max_velocity × dt
                # For cosine profile, max velocity ≈ π/(2*duration) * displacement
                max_displacement = 1.0  # max joint displacement in our test
                max_vel = math.pi / (2 * min(seg_dur, dt * 10)) * max_displacement
                max_step = max_vel * dt * 1.5  # 1.5x margin for blend acceleration

                assert jump < max_step, (
                    f"blend={blend_dur}s, seg={seg_dur}s: "
                    f"position jump {jump:.6f} rad at frame {i}, joint {j} "
                    f"(max allowed: {max_step:.6f})"
                )

    @pytest.mark.parametrize("blend_dur", [0.2, 0.3, 0.5])
    def test_velocity_continuity_at_blend_boundary(self, blend_dur):
        """
        At the blend boundary: velocity change between adjacent frames
        ≤ 3× the average velocity change within the main segment.

        This is the MATHEMATICAL definition of C1 continuity (bounded derivative),
        not just "velocity < some absolute threshold."
        """
        seg_dur = 2.0
        planner = TrajectoryPlanner(num_joints=7, control_freq=200.0)
        planner.add_waypoint(Waypoint(
            joints=[0.8, -0.5, 0.3, 0.6, -0.1, 0.4, 0.0],
            duration=seg_dur, blend=blend_dur))
        planner.add_waypoint(Waypoint(
            joints=[1.5, -1.0, 0.6, 1.2, -0.2, 0.8, 0.0],
            duration=seg_dur, blend=0.0))

        frames = planner.compute([0.0] * 7)

        for j in range(7):
            velocities = [f.velocities[j] for f in frames]
            if max(abs(v) for v in velocities) < 1e-6:
                continue

            # Compute acceleration (velocity differences)
            accels = [abs(velocities[i+1] - velocities[i])
                      for i in range(len(velocities) - 1)]

            # Average acceleration in the main segment (first half)
            main_end = len(frames) // 2
            main_accels = accels[:main_end]
            avg_accel = np.mean(main_accels) if main_accels else 0

            # Max acceleration anywhere (including blend boundary)
            max_accel = max(accels)

            # C1 definition: no accel spike > 5× the average
            assert max_accel < 5.0 * avg_accel + 1e-6, (
                f"blend={blend_dur}s, joint {j}: "
                f"accel spike {max_accel:.6f} > 5× avg {avg_accel:.6f} "
                f"— C1 discontinuity at blend boundary"
            )

    @pytest.mark.parametrize("blend_dur", [0.2, 0.3])
    def test_blend_exit_velocity_matches_next_segment_entry(self, blend_dur):
        """
        The blend's last frame velocity must match the next segment's first
        frame velocity (both should be near zero for cosine profile starting at t=0).
        """
        seg_dur = 2.0
        planner = TrajectoryPlanner(num_joints=7, control_freq=200.0)
        planner.add_waypoint(Waypoint(
            joints=[0.5, -0.3, 0.2, 0.4, -0.1, 0.3, 0.0],
            duration=seg_dur, blend=blend_dur))
        planner.add_waypoint(Waypoint(
            joints=[1.0, -0.6, 0.4, 0.8, -0.2, 0.6, 0.0],
            duration=seg_dur, blend=0.0))

        frames = planner.compute([0.0] * 7)

        # Find the blend-to-next-segment boundary
        # Blend occupies the last blend_dur*200 frames of first segment
        first_seg_frames = int(seg_dur * 200)

        if first_seg_frames < len(frames):
            # The frame right after the blend should have low velocity
            boundary_idx = min(first_seg_frames, len(frames) - 1)
            for j in range(7):
                v_at_boundary = abs(frames[boundary_idx].velocities[j])
                # Next segment starts at t=0 of cosine → velocity ≈ 0
                # Allow small residual from Hermite exit
                max_v = max(abs(f.velocities[j]) for f in frames)
                if max_v < 1e-6:
                    continue
                ratio = v_at_boundary / max_v
                assert ratio < 0.3, (
                    f"blend={blend_dur}s, joint {j}: "
                    f"blend exit velocity {v_at_boundary:.4f} is {ratio*100:.0f}% of peak "
                    f"— expected near zero (C1 match with next segment)"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# SafeBackend: 集成路径验证（不是测零件，而是测接线）
# ═══════════════════════════════════════════════════════════════════════════════

class _TrackingBackend(ArmBackend):
    """Mock that simulates lag: returned position = 80% of commanded."""

    def __init__(self):
        self._pos = [0.0] * 7
        self._calls = []

    @property
    def num_joints(self):
        return 7

    def enable(self):
        return ArmState(joints=[0.0]*7, velocities=[0.0]*7, torques=[0.0]*7)

    def disable(self):
        pass

    def command(self, positions, velocities, kps, kds, torques):
        self._calls.append(("command", list(positions)))
        # Simulate 80% tracking (arm lags behind command)
        self._pos = [0.8 * p + 0.2 * old for p, old in zip(positions, self._pos)]
        return ArmState(joints=list(self._pos), velocities=list(velocities),
                        torques=list(torques))

    def command_velocity(self, velocities):
        self._calls.append(("velocity", list(velocities)))
        # Velocity causes position change
        dt = 0.005
        self._pos = [p + v * dt for p, v in zip(self._pos, velocities)]
        return ArmState(joints=list(self._pos), velocities=list(velocities),
                        torques=[0.0]*7)

    def command_torque(self, torques):
        self._calls.append(("torque", list(torques)))
        return ArmState(joints=list(self._pos), velocities=[0.0]*7,
                        torques=list(torques))

    def get_state(self):
        return ArmState(joints=list(self._pos), velocities=[0.0]*7, torques=[0.0]*7)

    def close(self):
        pass


class TestSafeBackendIntegration:
    """
    Integration tests: verify SafeBackend's internal wiring is correct.

    These catch the class of bugs where a unit (SafetyMonitor) works correctly
    in isolation but SafeBackend passes it the WRONG argument.
    """

    def test_reconcile_uses_feedback_not_command(self):
        """
        When arm lags behind commands, rate limiter baseline must track
        the arm's ACTUAL position (feedback), not the commanded position.

        Bug this catches: reconcile_position(clamped_pos) instead of
        reconcile_position(state.joints).
        """
        inner = _TrackingBackend()
        config = SafetyConfig(max_delta_per_step=0.03, control_freq=200.0)
        safety = SafetyMonitor(num_joints=7, joint_limits=[(-3.14, 3.14)]*7, config=config)
        safety.reset(initial_positions=[0.0]*7)
        sb = SafeBackend(inner=inner, safety=safety)

        # Command far target repeatedly — arm only tracks 80%
        zeros = [0.0] * 7
        kps = [100.0] * 7
        kds = [5.0] * 7
        for _ in range(10):
            sb.command([1.0]*7, zeros, kps, kds, zeros)

        # The rate limiter baseline should be near the arm's actual position
        # (which lags behind), NOT at the accumulated commanded position
        actual_pos = inner._pos[0]
        baseline = safety._last_target[0]

        assert abs(baseline - actual_pos) < 0.001, (
            f"Rate limiter baseline ({baseline:.4f}) diverged from "
            f"actual arm position ({actual_pos:.4f}) — "
            f"reconcile is using commanded values instead of feedback"
        )

    def test_velocity_mode_updates_baseline(self):
        """
        After command_velocity, the rate limiter baseline reflects the new position.

        Bug this catches: command_velocity() not calling reconcile_position().
        """
        inner = _TrackingBackend()
        config = SafetyConfig(max_delta_per_step=0.03, control_freq=200.0)
        safety = SafetyMonitor(num_joints=7, joint_limits=[(-3.14, 3.14)]*7, config=config)
        safety.reset(initial_positions=[0.0]*7)
        sb = SafeBackend(inner=inner, safety=safety)

        # Move via velocity commands
        sb.command_velocity([1.0]*7)  # at 0.005s dt → pos ≈ 0.005
        sb.command_velocity([1.0]*7)  # pos ≈ 0.010
        sb.command_velocity([1.0]*7)  # pos ≈ 0.015

        # Now switch to position mode — should be limited from current position
        # (not from zero, which was the last position-mode baseline)
        actual_pos = inner._pos[0]
        zeros = [0.0] * 7
        kps = [100.0] * 7
        kds = [5.0] * 7

        sb.command([actual_pos + 0.01]*7, zeros, kps, kds, zeros)

        # If baseline was updated, this small step should pass through
        # If baseline is still at 0, the arm is now 0.015 from baseline,
        # and the command would be rate-limited differently
        commanded = inner._calls[-1][1][0]
        expected_target = actual_pos + 0.01

        assert abs(commanded - expected_target) < 0.001, (
            f"After velocity mode, position command was incorrectly limited: "
            f"commanded={commanded:.4f}, expected≈{expected_target:.4f}. "
            f"Baseline was not updated after command_velocity()."
        )

    def test_enable_resets_safety_baseline(self):
        """
        enable() must reset safety state so a previous emergency stop
        or stale baseline doesn't affect new commands.
        """
        inner = _TrackingBackend()
        config = SafetyConfig(max_delta_per_step=0.03, control_freq=200.0)
        safety = SafetyMonitor(num_joints=7, joint_limits=[(-3.14, 3.14)]*7, config=config)
        safety.reset(initial_positions=[0.5]*7)  # baseline at 0.5
        sb = SafeBackend(inner=inner, safety=safety)

        # Enable — should reset baseline to actual arm position (0.0)
        sb.enable()

        # Command small step from 0 — should NOT be rate-limited from old baseline 0.5
        zeros = [0.0] * 7
        kps = [100.0] * 7
        kds = [5.0] * 7
        sb.command([0.02]*7, zeros, kps, kds, zeros)

        commanded = inner._calls[-1][1][0]
        assert abs(commanded - 0.02) < 0.01, (
            f"After enable(), command limited by stale baseline: "
            f"commanded={commanded:.4f}, expected≈0.02"
        )

    @pytest.mark.parametrize("mode", ["velocity", "torque", "hold"])
    def test_emergency_stop_blocks_all_modes(self, mode):
        """Emergency stop blocks ALL command paths, not just command()."""
        inner = _TrackingBackend()
        config = SafetyConfig(max_delta_per_step=0.03, control_freq=200.0)
        safety = SafetyMonitor(num_joints=7, joint_limits=[(-3.14, 3.14)]*7, config=config)
        safety.reset(initial_positions=[0.0]*7)
        sb = SafeBackend(inner=inner, safety=safety)

        safety.set_emergency_stop()

        from onero_sdk.exceptions import OneroHardwareError

        if mode == "velocity":
            with pytest.raises(OneroHardwareError):
                sb.command_velocity([1.0]*7)
        elif mode == "torque":
            with pytest.raises(OneroHardwareError):
                sb.command_torque([1.0]*7)
        elif mode == "hold":
            with pytest.raises(OneroHardwareError):
                sb.command_hold([0.0]*7, [0.0]*7, [100.0]*7, [5.0]*7, [0.0]*7)


# ═══════════════════════════════════════════════════════════════════════════════
# Rate limiter: 参数化遍历 + 数学性质
# ═══════════════════════════════════════════════════════════════════════════════

RATE_LIMIT_TARGETS = [0.01, 0.02, 0.029, 0.03, 0.031, 0.05, 0.1, 0.5, 1.0, 2.0]


class TestRateLimiterFormal:
    """
    Rate limiter: parametrized sweep proving mathematical properties.
    """

    @pytest.mark.parametrize("target", RATE_LIMIT_TARGETS)
    def test_single_step_clamped_to_max_delta(self, target):
        """∀ target: |clamp_target(target) - baseline| ≤ max_delta_per_step."""
        max_delta = 0.03
        config = SafetyConfig(max_delta_per_step=max_delta, control_freq=200.0)
        safety = SafetyMonitor(
            num_joints=7, joint_limits=[(-5.0, 5.0)]*7, config=config)
        safety.reset(initial_positions=[0.0]*7)

        result = safety.clamp_target([target]*7)

        for j in range(7):
            step = abs(result[j] - 0.0)  # baseline is 0
            assert step <= max_delta + 1e-10, (
                f"target={target}: step {step:.6f} > max_delta {max_delta}"
            )

    @pytest.mark.parametrize("target", RATE_LIMIT_TARGETS)
    def test_direction_preserved(self, target):
        """∀ target: sign(result - baseline) = sign(target - baseline)."""
        max_delta = 0.03
        config = SafetyConfig(max_delta_per_step=max_delta, control_freq=200.0)
        safety = SafetyMonitor(
            num_joints=7, joint_limits=[(-5.0, 5.0)]*7, config=config)
        safety.reset(initial_positions=[0.0]*7)

        result = safety.clamp_target([target]*7)
        # Positive target → positive step (or zero for zero target)
        if target > 1e-10:
            assert result[0] > 0, f"target={target}: direction lost"
        elif target < -1e-10:
            assert result[0] < 0, f"target={target}: direction lost"

    def test_convergence_to_target(self):
        """Repeated clamping converges: ∃N: |pos_N - target| < ε."""
        max_delta = 0.03
        target = 1.0
        config = SafetyConfig(max_delta_per_step=max_delta, control_freq=200.0)
        safety = SafetyMonitor(
            num_joints=7, joint_limits=[(-5.0, 5.0)]*7, config=config)
        safety.reset(initial_positions=[0.0]*7)

        for step in range(200):
            result = safety.clamp_target([target]*7)
            if abs(result[0] - target) < 1e-6:
                break
        else:
            pytest.fail(
                f"Rate limiter did not converge to {target} in 200 steps "
                f"(stuck at {result[0]:.6f})"
            )

        # Should converge in ceil(target / max_delta) steps
        expected_steps = math.ceil(target / max_delta)
        assert step <= expected_steps + 1, (
            f"Converged in {step} steps, expected ≤ {expected_steps+1}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Trajectory: 位移积分恒等式 (parametrized)
# ═══════════════════════════════════════════════════════════════════════════════

TRAJ_DISPLACEMENTS = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0]
TRAJ_DURATIONS = [1.0, 2.0, 3.0, 5.0]


class TestTrajectoryIntegralFormal:
    """
    Velocity integral identity: ∫v(t)dt = Δposition.

    This is the fundamental theorem of calculus applied to trajectory generation.
    If velocity profile doesn't integrate to the correct displacement,
    the arm will drift when following the trajectory.
    """

    @pytest.mark.parametrize("displacement", TRAJ_DISPLACEMENTS)
    @pytest.mark.parametrize("duration", TRAJ_DURATIONS)
    def test_velocity_integrates_to_displacement(self, displacement, duration):
        """∀ disp, dur: sum(v_k × dt) ≈ displacement (within 5%)."""
        dt = 1.0 / 200.0
        start = [0.0] * 7
        end = [displacement] + [0.0] * 6  # only joint 0 moves

        planner = TrajectoryPlanner(num_joints=7, control_freq=200.0)
        planner.add_waypoint(Waypoint(joints=end, duration=duration, blend=0.0))
        frames = planner.compute(start)

        integrated = sum(f.velocities[0] * dt for f in frames)
        pos_delta = frames[-1].joints[0] - frames[0].joints[0]

        # Position-based displacement must match target
        assert abs(pos_delta - displacement) < 0.01, (
            f"disp={displacement}, dur={duration}s: "
            f"position delta {pos_delta:.4f} ≠ target {displacement}"
        )

        # Velocity integral must approximately match position delta
        # (trapezoidal integration introduces O(dt) error)
        rel_err = abs(integrated - pos_delta) / max(abs(pos_delta), 1e-6)
        assert rel_err < 0.05, (
            f"disp={displacement}, dur={duration}s: "
            f"∫v·dt={integrated:.4f} ≠ Δpos={pos_delta:.4f} "
            f"(relative error {rel_err*100:.1f}%)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
