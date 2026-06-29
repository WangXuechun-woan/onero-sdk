"""
Robustness tests — invalid, illegal, and edge-case inputs.

Design philosophy:
- Safety-critical system MUST handle garbage gracefully
- NaN, Inf, negative, out-of-range → must NOT propagate to hardware
- Either reject (raise) or clamp to safe values — never silently forward
- Tests cover: NaN, ±Inf, empty, wrong-length, huge, tiny, negative-where-positive-expected

These tests verify that the SDK protects hardware from software bugs upstream.
"""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from onero_sdk.geometry import slerp_matrix, rotvec_to_matrix, signed_arc_angle, fit_circle_3points
from onero_sdk.trajectory import TrajectoryPlanner, Waypoint
from onero_sdk.safety import SafetyMonitor, SafetyConfig
from onero_sdk.safe_backend import SafeBackend
from onero_sdk.backend import ArmBackend, ArmState


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: values that should NEVER reach hardware
# ═══════════════════════════════════════════════════════════════════════════════

INVALID_SCALARS = [
    float('nan'),
    float('inf'),
    float('-inf'),
    1e308,        # near max float
    -1e308,       # near min float
    1e-308,       # subnormal territory
]

INVALID_POSITIONS_7DOF = [
    [float('inf')] * 7,
    [float('-inf')] * 7,
    [1e10] * 7,             # absurdly large
    [-1e10] * 7,            # absurdly negative
    [100.0] * 7,            # far beyond joint limits (±3.14)
    [-100.0] * 7,           # far negative beyond joint limits
]

# NaN is special — np.clip propagates NaN unlike Inf
NAN_POSITIONS_7DOF = [
    [float('nan')] * 7,
]


# ═══════════════════════════════════════════════════════════════════════════════
# SLERP robustness: invalid rotation inputs
# ═══════════════════════════════════════════════════════════════════════════════

class TestSLERPRobustness:
    """SLERP must produce finite, valid SO(3) or raise — never NaN/Inf silently."""

    @pytest.mark.parametrize("t", [-1.0, -0.5, 1.5, 2.0, 100.0, -100.0])
    def test_slerp_t_out_of_range(self, t):
        """t outside [0,1]: result must still be valid SO(3) (extrapolation) or raise."""
        R0 = rotvec_to_matrix([0.3, -0.2, 0.5])
        R1 = R0 @ rotvec_to_matrix([0, 0, 0.5])
        try:
            R = slerp_matrix(R0, R1, t)
            # If it returns, must be valid rotation
            assert np.all(np.isfinite(R)), f"t={t}: NaN/Inf in result"
            det = np.linalg.det(R)
            assert abs(det - 1.0) < 1e-4 or abs(det + 1.0) < 1e-4, (
                f"t={t}: det={det}, not a valid rotation"
            )
        except (ValueError, np.linalg.LinAlgError):
            pass  # raising is acceptable

    @pytest.mark.parametrize("t", [float('nan'), float('inf'), float('-inf')])
    def test_slerp_t_invalid_special(self, t):
        """t = NaN/Inf: must raise or return identity — never produce NaN silently."""
        R0 = rotvec_to_matrix([0.3, -0.2, 0.5])
        R1 = R0 @ rotvec_to_matrix([0, 0, 0.5])
        try:
            R = slerp_matrix(R0, R1, t)
            # If it doesn't raise, NaN should NOT be in the result
            if np.any(np.isnan(R)):
                pytest.fail(f"t={t}: NaN propagated silently into rotation matrix")
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            pass  # raising is acceptable

    def test_slerp_non_orthogonal_input(self):
        """Non-SO(3) input: must not crash or produce garbage silently."""
        R0 = np.eye(3)
        R_bad = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)  # det=2
        try:
            R = slerp_matrix(R0, R_bad, 0.5)
            assert np.all(np.isfinite(R)), "NaN from non-orthogonal input"
        except (ValueError, np.linalg.LinAlgError):
            pass

    def test_slerp_nan_in_matrix(self):
        """NaN in input matrix: must not propagate silently."""
        R0 = np.eye(3)
        R_nan = np.eye(3)
        R_nan[0, 0] = float('nan')
        try:
            R = slerp_matrix(R0, R_nan, 0.5)
            if np.any(np.isnan(R)):
                pytest.fail("NaN in input matrix propagated to output without detection")
        except (ValueError, FloatingPointError):
            pass

    def test_slerp_zero_matrix(self):
        """All-zeros matrix: must handle gracefully."""
        R0 = np.eye(3)
        R_zero = np.zeros((3, 3))
        try:
            R = slerp_matrix(R0, R_zero, 0.5)
            assert np.all(np.isfinite(R)), "Zero-matrix input produced non-finite output"
        except (ValueError, ZeroDivisionError, np.linalg.LinAlgError):
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# Circle fitting: degenerate geometry
# ═══════════════════════════════════════════════════════════════════════════════

class TestCircleFitRobustness:
    """fit_circle_3points with degenerate inputs."""

    def test_identical_points(self):
        """Three identical points: must not crash."""
        p = np.array([1.0, 2.0, 3.0])
        try:
            center, radius, normal = fit_circle_3points(p, p, p)
            assert np.all(np.isfinite(center))
            assert np.isfinite(radius)
            assert np.all(np.isfinite(normal))
        except (ValueError, ZeroDivisionError):
            pass

    def test_two_identical_one_different(self):
        """Two identical + one different: degenerate circle."""
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        try:
            center, radius, normal = fit_circle_3points(p0, p1, p2)
            assert np.all(np.isfinite(center))
            assert np.isfinite(radius)
        except (ValueError, ZeroDivisionError):
            pass

    @pytest.mark.parametrize("scale", [1e-15, 1e-10, 1e10, 1e15])
    def test_extreme_scale(self, scale):
        """Points at extreme scales: numerical stability."""
        p0 = np.array([0, 0, 0.0]) * scale
        p1 = np.array([1, 0, 0.0]) * scale
        p2 = np.array([0.5, 0.866, 0.0]) * scale
        # These 3 points form a triangle with circumradius ≈ 0.5774 × scale
        expected_radius = scale / math.sqrt(3)
        try:
            center, radius, normal = fit_circle_3points(p0, p1, p2)
            assert np.all(np.isfinite(center)), f"scale={scale}: non-finite center"
            assert np.isfinite(radius), f"scale={scale}: non-finite radius"
            if 1e-8 < scale < 1e8:  # extreme scales lose double precision
                assert abs(radius - expected_radius) / expected_radius < 0.01, (
                    f"scale={scale}: radius={radius}, expected≈{expected_radius}"
                )
        except (ValueError, np.linalg.LinAlgError):
            pass

    def test_nan_in_points(self):
        """NaN in point coordinates: must not propagate."""
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([1.0, 0.0, 0.0])
        p_nan = np.array([float('nan'), 0.0, 0.0])
        try:
            center, radius, normal = fit_circle_3points(p0, p1, p_nan)
            if np.any(np.isnan(center)):
                pytest.fail("NaN propagated from input point to center")
        except (ValueError, FloatingPointError):
            pass

    def test_inf_in_points(self):
        """Inf in point coordinates."""
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([1.0, 0.0, 0.0])
        p_inf = np.array([float('inf'), 0.0, 0.0])
        try:
            center, radius, normal = fit_circle_3points(p0, p1, p_inf)
            assert np.all(np.isfinite(center)), "Inf propagated to center"
        except (ValueError, FloatingPointError, OverflowError):
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# signed_arc_angle: degenerate vectors
# ═══════════════════════════════════════════════════════════════════════════════

class TestArcAngleRobustness:
    """signed_arc_angle with degenerate/invalid inputs."""

    def test_zero_length_vectors(self):
        """Zero-length vectors: must not produce NaN."""
        normal = np.array([0, 0, 1.0])
        zero = np.array([0, 0, 0.0])
        v1 = np.array([1, 0, 0.0])
        try:
            angle = signed_arc_angle(zero, v1, normal, v1)
            assert np.isfinite(angle), "Zero-length v0 produced non-finite angle"
        except (ValueError, ZeroDivisionError):
            pass

    def test_parallel_vectors(self):
        """v0 parallel to v2 (angle = 0 or π): determinate result."""
        normal = np.array([0, 0, 1.0])
        v0 = np.array([1, 0, 0.0])
        v2 = np.array([1, 0, 0.0])  # same as v0
        v1 = np.array([0, 1, 0.0])  # via at 90°
        angle = signed_arc_angle(v0, v2, normal, v1)
        assert np.isfinite(angle)
        # Full circle expected
        assert abs(abs(angle) - 2 * math.pi) < 0.01

    def test_antiparallel_vectors(self):
        """v0 antiparallel to v2: half circle."""
        normal = np.array([0, 0, 1.0])
        v0 = np.array([1, 0, 0.0])
        v2 = np.array([-1, 0, 0.0])
        v1 = np.array([0, 1, 0.0])
        angle = signed_arc_angle(v0, v2, normal, v1)
        assert np.isfinite(angle)
        assert abs(angle - math.pi) < 0.01

    def test_nan_in_vectors(self):
        """NaN in input vectors."""
        normal = np.array([0, 0, 1.0])
        v0 = np.array([float('nan'), 0, 0.0])
        v2 = np.array([1, 0, 0.0])
        v1 = np.array([0, 1, 0.0])
        try:
            angle = signed_arc_angle(v0, v2, normal, v1)
            if np.isnan(angle):
                pytest.fail("NaN propagated from input to arc angle")
        except (ValueError, FloatingPointError):
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# SafetyMonitor: invalid command inputs MUST be caught
# ═══════════════════════════════════════════════════════════════════════════════

class TestSafetyInvalidInputs:
    """
    SafetyMonitor MUST NOT forward NaN/Inf to hardware.

    This is the last line of defense — if invalid values reach clamp_target()
    and pass through unchanged, the physical arm receives garbage commands.
    """

    def _make_safety(self):
        config = SafetyConfig(max_delta_per_step=0.03, max_velocity=5.0,
                              control_freq=200.0)
        safety = SafetyMonitor(
            num_joints=7, joint_limits=[(-3.14, 3.14)]*7, config=config)
        safety.reset(initial_positions=[0.0]*7)
        return safety

    @pytest.mark.parametrize("invalid_pos", INVALID_POSITIONS_7DOF)
    def test_clamp_target_rejects_or_clamps_invalid(self, invalid_pos):
        """
        ∀ invalid input: clamp_target produces only finite values within limits.

        This is CRITICAL: NaN/Inf reaching the motor controller = hardware damage.
        """
        safety = self._make_safety()
        try:
            result = safety.clamp_target(invalid_pos)
            for j in range(7):
                assert np.isfinite(result[j]), (
                    f"Joint {j}: non-finite value {result[j]} passed through "
                    f"safety layer! Input was {invalid_pos[j]}"
                )
                assert -3.14 <= result[j] <= 3.14, (
                    f"Joint {j}: value {result[j]} outside joint limits "
                    f"after clamping! Input was {invalid_pos[j]}"
                )
        except (ValueError, TypeError):
            pass  # raising is also acceptable

    def test_clamp_target_nan_must_not_propagate(self):
        """NaN input MUST NOT reach hardware — this is the most critical safety invariant."""
        safety = self._make_safety()
        result = safety.clamp_target([float('nan')] * 7)
        for j in range(7):
            assert np.isfinite(result[j]), (
                f"Joint {j}: NaN passed through safety layer to hardware!"
            )

    @pytest.mark.parametrize("invalid_vel", [
        [float('nan')] * 7,
        [float('inf')] * 7,
        [float('-inf')] * 7,
        [1e10] * 7,
        [-1e10] * 7,
    ])
    def test_clamp_velocity_rejects_or_clamps_invalid(self, invalid_vel):
        """∀ invalid velocity: result is finite and within limits."""
        safety = self._make_safety()
        try:
            result = safety.clamp_velocity(invalid_vel)
            for j in range(7):
                assert np.isfinite(result[j]), (
                    f"Joint {j}: non-finite velocity {result[j]} passed through! "
                    f"Input was {invalid_vel[j]}"
                )
                assert abs(result[j]) <= 5.0 + 1e-10, (
                    f"Joint {j}: velocity {result[j]} exceeds max after clamping"
                )
        except (ValueError, TypeError):
            pass

    def test_reconcile_with_nan_position(self):
        """reconcile_position with NaN must not corrupt baseline."""
        safety = self._make_safety()
        # Normal command first
        safety.clamp_target([0.1]*7)

        # Reconcile with NaN — should NOT corrupt internal state
        try:
            safety.reconcile_position([float('nan')]*7)
        except (ValueError, TypeError):
            pass  # acceptable to raise

        # Next clamp_target must still produce finite results
        result = safety.clamp_target([0.2]*7)
        for j in range(7):
            if not np.isfinite(result[j]):
                pytest.fail(
                    f"Joint {j}: NaN reconcile corrupted baseline — "
                    f"subsequent clamp_target produces {result[j]}"
                )

    def test_reconcile_with_inf_position(self):
        """reconcile_position with Inf must not corrupt baseline."""
        safety = self._make_safety()
        safety.clamp_target([0.1]*7)
        try:
            safety.reconcile_position([float('inf')]*7)
        except (ValueError, TypeError):
            pass

        result = safety.clamp_target([0.2]*7)
        for j in range(7):
            if not np.isfinite(result[j]):
                pytest.fail(
                    f"Joint {j}: Inf reconcile corrupted baseline — "
                    f"subsequent clamp_target produces {result[j]}"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# SafeBackend: invalid inputs through the full stack
# ═══════════════════════════════════════════════════════════════════════════════

class _RecordingBackend(ArmBackend):
    """Records exactly what reaches the inner backend."""

    def __init__(self):
        self.received_positions = []
        self.received_velocities = []
        self.received_torques = []

    @property
    def num_joints(self):
        return 7

    def enable(self):
        return ArmState(joints=[0.0]*7, velocities=[0.0]*7, torques=[0.0]*7)

    def disable(self):
        pass

    def command(self, positions, velocities, kps, kds, torques):
        self.received_positions.append(list(positions))
        self.received_velocities.append(list(velocities))
        self.received_torques.append(list(torques))
        return ArmState(joints=list(positions), velocities=list(velocities),
                        torques=list(torques))

    def command_velocity(self, velocities):
        self.received_velocities.append(list(velocities))
        return ArmState(joints=[0.0]*7, velocities=list(velocities), torques=[0.0]*7)

    def command_torque(self, torques):
        self.received_torques.append(list(torques))
        return ArmState(joints=[0.0]*7, velocities=[0.0]*7, torques=list(torques))

    def get_state(self):
        return ArmState(joints=[0.0]*7, velocities=[0.0]*7, torques=[0.0]*7)

    def close(self):
        pass


class TestSafeBackendInvalidInputs:
    """
    Full-stack invalid input tests: verify nothing dangerous reaches hardware.

    The SafeBackend→SafetyMonitor→InnerBackend chain must guarantee:
    1. NaN never reaches the inner backend
    2. Inf never reaches the inner backend
    3. Values outside physical limits never reach the inner backend
    """

    def _make_safe_backend(self):
        inner = _RecordingBackend()
        config = SafetyConfig(max_delta_per_step=0.03, max_velocity=5.0,
                              control_freq=200.0)
        safety = SafetyMonitor(
            num_joints=7, joint_limits=[(-3.14, 3.14)]*7, config=config)
        safety.reset(initial_positions=[0.0]*7)
        sb = SafeBackend(inner=inner, safety=safety, torque_limits=[50.0]*7)
        return sb, inner

    @pytest.mark.parametrize("bad_pos", INVALID_POSITIONS_7DOF)
    def test_command_with_invalid_positions(self, bad_pos):
        """NaN/Inf positions must not reach inner backend."""
        sb, inner = self._make_safe_backend()
        zeros = [0.0] * 7
        kps = [100.0] * 7
        kds = [5.0] * 7
        try:
            sb.command(bad_pos, zeros, kps, kds, zeros)
            if inner.received_positions:
                for j in range(7):
                    val = inner.received_positions[-1][j]
                    assert np.isfinite(val), (
                        f"Joint {j}: non-finite position {val} reached hardware! "
                        f"Input was {bad_pos[j]}"
                    )
                    assert -3.14 <= val <= 3.14, (
                        f"Joint {j}: position {val} outside joint limits reached hardware!"
                    )
        except (ValueError, TypeError, OverflowError):
            pass  # raising before forwarding is acceptable

    def test_command_with_nan_positions(self):
        """NaN positions MUST NOT reach inner backend — safety critical."""
        sb, inner = self._make_safe_backend()
        zeros = [0.0] * 7
        kps = [100.0] * 7
        kds = [5.0] * 7
        sb.command([float('nan')]*7, zeros, kps, kds, zeros)
        if inner.received_positions:
            for j in range(7):
                val = inner.received_positions[-1][j]
                assert np.isfinite(val), (
                    f"Joint {j}: NaN reached hardware!"
                )

    @pytest.mark.parametrize("bad_vel", [
        [float('nan')] * 7,
        [float('inf')] * 7,
        [1e10] * 7,
    ])
    def test_command_velocity_with_invalid(self, bad_vel):
        """NaN/Inf velocities must not reach inner backend."""
        sb, inner = self._make_safe_backend()
        try:
            sb.command_velocity(bad_vel)
            if inner.received_velocities:
                for j in range(7):
                    val = inner.received_velocities[-1][j]
                    assert np.isfinite(val), (
                        f"Joint {j}: non-finite velocity {val} reached hardware!"
                    )
                    assert abs(val) <= 5.0 + 1e-6, (
                        f"Joint {j}: velocity {val} exceeds max_velocity reached hardware!"
                    )
        except (ValueError, TypeError, OverflowError):
            pass

    @pytest.mark.parametrize("bad_torque", [
        [float('nan')] * 7,
        [float('inf')] * 7,
        [1e10] * 7,
        [-1e10] * 7,
    ])
    def test_command_torque_with_invalid(self, bad_torque):
        """NaN/Inf torques must not reach inner backend."""
        sb, inner = self._make_safe_backend()
        try:
            sb.command_torque(bad_torque)
            if inner.received_torques:
                for j in range(7):
                    val = inner.received_torques[-1][j]
                    assert np.isfinite(val), (
                        f"Joint {j}: non-finite torque {val} reached hardware!"
                    )
                    assert abs(val) <= 50.0 + 1e-6, (
                        f"Joint {j}: torque {val} exceeds limit reached hardware!"
                    )
        except (ValueError, TypeError, OverflowError):
            pass

    def test_mixed_valid_and_nan(self):
        """Some joints valid, some NaN: valid joints must still be reasonable."""
        sb, inner = self._make_safe_backend()
        mixed = [0.01, 0.02, float('nan'), 0.01, float('inf'), 0.02, 0.01]
        zeros = [0.0] * 7
        kps = [100.0] * 7
        kds = [5.0] * 7
        try:
            sb.command(mixed, zeros, kps, kds, zeros)
            if inner.received_positions:
                for j in range(7):
                    val = inner.received_positions[-1][j]
                    assert np.isfinite(val), (
                        f"Joint {j}: non-finite {val} reached hardware (mixed input)"
                    )
        except (ValueError, TypeError):
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# Trajectory planner: invalid waypoints
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrajectoryInvalidInputs:
    """Trajectory planner must handle invalid waypoints gracefully."""

    @pytest.mark.parametrize("duration", [0.0, -1.0, float('nan'), float('inf')])
    def test_invalid_duration(self, duration):
        """Invalid duration: must raise or produce empty/finite trajectory."""
        planner = TrajectoryPlanner(num_joints=7, control_freq=200.0)
        try:
            planner.add_waypoint(Waypoint(
                joints=[0.5]*7, duration=duration, blend=0.0))
            frames = planner.compute([0.0]*7)
            # If it produces frames, they must be finite
            for f in frames:
                for j in range(7):
                    assert np.isfinite(f.joints[j]), (
                        f"Non-finite joint position from duration={duration}"
                    )
                    assert np.isfinite(f.velocities[j]), (
                        f"Non-finite velocity from duration={duration}"
                    )
        except (ValueError, ZeroDivisionError, OverflowError):
            pass  # raising is acceptable

    def test_nan_in_waypoint_joints(self):
        """NaN in waypoint joints: must not produce NaN trajectory."""
        planner = TrajectoryPlanner(num_joints=7, control_freq=200.0)
        try:
            planner.add_waypoint(Waypoint(
                joints=[float('nan')]*7, duration=1.0, blend=0.0))
            frames = planner.compute([0.0]*7)
            for f in frames:
                for j in range(7):
                    if np.isnan(f.joints[j]):
                        pytest.fail("NaN in waypoint propagated to trajectory frame")
        except (ValueError, TypeError):
            pass

    def test_inf_in_waypoint_joints(self):
        """Inf in waypoint joints."""
        planner = TrajectoryPlanner(num_joints=7, control_freq=200.0)
        try:
            planner.add_waypoint(Waypoint(
                joints=[float('inf')]*7, duration=1.0, blend=0.0))
            frames = planner.compute([0.0]*7)
            for f in frames:
                for j in range(7):
                    if not np.isfinite(f.joints[j]):
                        pytest.fail("Inf in waypoint propagated to trajectory frame")
        except (ValueError, OverflowError):
            pass

    @pytest.mark.parametrize("blend", [-0.5, float('nan'), float('inf')])
    def test_invalid_blend_duration(self, blend):
        """Invalid blend duration: graceful handling."""
        planner = TrajectoryPlanner(num_joints=7, control_freq=200.0)
        try:
            planner.add_waypoint(Waypoint(joints=[0.5]*7, duration=2.0, blend=blend))
            planner.add_waypoint(Waypoint(joints=[1.0]*7, duration=2.0, blend=0.0))
            frames = planner.compute([0.0]*7)
            for f in frames:
                for j in range(7):
                    assert np.isfinite(f.joints[j]), (
                        f"Non-finite from blend={blend}"
                    )
        except (ValueError, ZeroDivisionError, OverflowError):
            pass

    def test_blend_longer_than_segment(self):
        """Blend duration > segment duration: should not crash."""
        planner = TrajectoryPlanner(num_joints=7, control_freq=200.0)
        try:
            planner.add_waypoint(Waypoint(joints=[0.5]*7, duration=1.0, blend=2.0))
            planner.add_waypoint(Waypoint(joints=[1.0]*7, duration=1.0, blend=0.0))
            frames = planner.compute([0.0]*7)
            for f in frames:
                for j in range(7):
                    assert np.isfinite(f.joints[j])
        except (ValueError, ZeroDivisionError):
            pass

    def test_very_short_duration(self):
        """Duration shorter than one control step: edge case."""
        planner = TrajectoryPlanner(num_joints=7, control_freq=200.0)
        planner.add_waypoint(Waypoint(
            joints=[1.0]*7, duration=0.001, blend=0.0))  # 0.2 of a frame
        try:
            frames = planner.compute([0.0]*7)
            # Should produce at least 1 frame or raise
            if frames:
                for f in frames:
                    for j in range(7):
                        assert np.isfinite(f.joints[j])
        except (ValueError, ZeroDivisionError):
            pass

    def test_empty_start_joints(self):
        """Empty start_joints list."""
        planner = TrajectoryPlanner(num_joints=7, control_freq=200.0)
        planner.add_waypoint(Waypoint(joints=[1.0]*7, duration=1.0, blend=0.0))
        try:
            frames = planner.compute([])  # wrong length
            # If it doesn't raise, check finiteness
        except (ValueError, IndexError, TypeError):
            pass  # expected

    def test_extremely_high_control_freq(self):
        """Very high control_freq: many frames but no crash."""
        planner = TrajectoryPlanner(num_joints=7, control_freq=100000.0)
        planner.add_waypoint(Waypoint(joints=[0.5]*7, duration=0.01, blend=0.0))
        frames = planner.compute([0.0]*7)
        # Should produce 1000 frames (100000 Hz × 0.01s)
        assert len(frames) > 0
        for f in frames[:10]:  # spot check
            for j in range(7):
                assert np.isfinite(f.joints[j])


# ═══════════════════════════════════════════════════════════════════════════════
# rotvec_to_matrix: edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestRotvecRobustness:
    """rotvec_to_matrix with edge-case rotation vectors."""

    @pytest.mark.parametrize("angle", [0, 1e-15, 1e-10, 1e-5, 0.001, 0.01])
    def test_very_small_angles(self, angle):
        """Near-zero rotation: must return near-identity, never NaN."""
        R = rotvec_to_matrix([angle, 0, 0])
        assert np.all(np.isfinite(R))
        # Frobenius norm of (R - I) for rotation θ ≈ √2 × θ
        assert np.linalg.norm(R - np.eye(3)) < 2 * angle + 1e-10

    @pytest.mark.parametrize("angle", [math.pi, 2*math.pi, 3*math.pi, 10*math.pi])
    def test_large_angles(self, angle):
        """Large rotation angles (multiple wraps): valid SO(3)."""
        R = rotvec_to_matrix([0, 0, angle])
        assert np.all(np.isfinite(R))
        det = np.linalg.det(R)
        assert abs(det - 1.0) < 1e-8, f"angle={angle}: det={det}"
        orth = np.linalg.norm(R.T @ R - np.eye(3))
        assert orth < 1e-8, f"angle={angle}: orthogonality error={orth}"

    def test_zero_vector(self):
        """Zero rotation vector: identity matrix."""
        R = rotvec_to_matrix([0, 0, 0])
        assert np.allclose(R, np.eye(3), atol=1e-15)

    @pytest.mark.parametrize("val", [float('nan'), float('inf'), float('-inf')])
    def test_invalid_rotvec_values(self, val):
        """NaN/Inf in rotation vector raises error."""
        with pytest.raises(Exception):
            rotvec_to_matrix([val, 0, 0])

    @pytest.mark.parametrize("scale", [1e-20, 1e-10, 1e10, 1e20])
    def test_extreme_magnitude_axis(self, scale):
        """Rotation vector with extreme magnitudes."""
        try:
            R = rotvec_to_matrix([scale, 0, 0])
            assert np.all(np.isfinite(R)), f"scale={scale}: non-finite output"
            det = np.linalg.det(R)
            assert abs(det - 1.0) < 1e-4, f"scale={scale}: det={det}"
        except (ValueError, OverflowError, FloatingPointError):
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# SafetyConfig: invalid configuration parameters
# ═══════════════════════════════════════════════════════════════════════════════

class TestSafetyConfigEdgeCases:
    """SafetyConfig with unusual but possible configurations."""

    @pytest.mark.parametrize("max_delta", [0.0, -0.01, 1e-15])
    def test_zero_or_negative_max_delta(self, max_delta):
        """Zero/negative max_delta_per_step: must not allow movement or crash."""
        config = SafetyConfig(max_delta_per_step=max_delta, control_freq=200.0)
        safety = SafetyMonitor(
            num_joints=7, joint_limits=[(-3.14, 3.14)]*7, config=config)
        safety.reset(initial_positions=[0.0]*7)
        try:
            result = safety.clamp_target([1.0]*7)
            # With zero max_delta, should stay at baseline
            for j in range(7):
                assert abs(result[j]) <= abs(max_delta) + 1e-10, (
                    f"max_delta={max_delta}: joint moved {result[j]} from baseline"
                )
        except (ValueError, ZeroDivisionError):
            pass

    @pytest.mark.parametrize("freq", [0.0, -100.0, 1e-10, 1e6])
    def test_extreme_control_freq(self, freq):
        """Extreme control_freq values."""
        try:
            config = SafetyConfig(max_delta_per_step=0.03, control_freq=freq)
            safety = SafetyMonitor(
                num_joints=7, joint_limits=[(-3.14, 3.14)]*7, config=config)
            safety.reset(initial_positions=[0.0]*7)
            rate = safety.max_rate
            assert np.isfinite(rate), f"freq={freq}: max_rate is not finite ({rate})"
        except (ValueError, ZeroDivisionError):
            pass

    def test_inverted_joint_limits(self):
        """Joint limits where min > max: clamp behavior."""
        config = SafetyConfig(max_delta_per_step=0.03)
        # Intentionally inverted limits
        inverted_limits = [(3.14, -3.14)] * 7
        safety = SafetyMonitor(
            num_joints=7, joint_limits=inverted_limits, config=config)
        safety.reset(initial_positions=[0.0]*7)
        try:
            result = safety.clamp_target([0.5]*7)
            # np.clip with min>max produces undefined behavior in older numpy
            # but should not crash
            for j in range(7):
                assert np.isfinite(result[j])
        except (ValueError, TypeError):
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# Force-mode input validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestForceModeInputValidation:
    """force_mode must reject NaN/Inf and malformed inputs before they reach backend."""

    @pytest.fixture
    def arm(self):
        try:
            from onero_sdk import Arm
            from onero_sdk.mujoco_backend import MujocoBackend
        except ImportError:
            pytest.skip("MuJoCo not installed")
        backend = MujocoBackend(model_name="A1R")
        a = Arm("A1R", backend=backend)
        a.enable()
        a.movej([0.0, -0.3, 0.2, -0.5, 0.0, 0.3, 0.0], duration=1.0)
        yield a
        a.disable()
        backend.close()

    def test_selection_nan_rejected(self, arm):
        with pytest.raises(Exception, match="NaN or Inf"):
            arm.force_mode(selection=[0, 0, float('nan'), 0, 0, 0],
                           wrench=[0, 0, -5, 0, 0, 0])

    def test_selection_wrong_length_rejected(self, arm):
        with pytest.raises(Exception, match="6-element"):
            arm.force_mode(selection=[0, 0, 1], wrench=[0, 0, -5, 0, 0, 0])

    def test_selection_non_binary_rejected(self, arm):
        with pytest.raises(Exception, match="0 or 1"):
            arm.force_mode(selection=[0, 0, 0.5, 0, 0, 0],
                           wrench=[0, 0, -5, 0, 0, 0])

    def test_wrench_nan_rejected(self, arm):
        with pytest.raises(Exception, match="NaN or Inf"):
            arm.force_mode(selection=[0, 0, 1, 0, 0, 0],
                           wrench=[0, 0, float('inf'), 0, 0, 0])

    def test_wrench_wrong_length_rejected(self, arm):
        with pytest.raises(Exception, match="6-element"):
            arm.force_mode(selection=[0, 0, 1, 0, 0, 0],
                           wrench=[0, 0, -5])

    def test_limits_nan_rejected(self, arm):
        with pytest.raises(Exception, match="NaN or Inf"):
            arm.force_mode(selection=[0, 0, 1, 0, 0, 0],
                           wrench=[0, 0, -5, 0, 0, 0],
                           limits=[0.1, 0.1, float('nan'), 0.1, 0.1, 0.1])

    def test_limits_negative_rejected(self, arm):
        with pytest.raises(Exception, match="non-negative"):
            arm.force_mode(selection=[0, 0, 1, 0, 0, 0],
                           wrench=[0, 0, -5, 0, 0, 0],
                           limits=[0.1, 0.1, -0.5, 0.1, 0.1, 0.1])

    def test_valid_inputs_accepted(self, arm):
        state = arm.force_mode(selection=[0, 0, 1, 0, 0, 0],
                               wrench=[0, 0, -5, 0, 0, 0],
                               limits=[0.5, 0.5, 0.5, 1.0, 1.0, 1.0])
        assert state is not None


# ═══════════════════════════════════════════════════════════════════════════════
# DualArm.create() input validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestDualArmCreateValidation:
    """DualArm.create() must reject dangerous configuration (world_frame without spacing)."""

    def test_no_spacing_no_position_raises(self):
        from onero_sdk import DualArm
        from onero_sdk.exceptions import OneroConfigError
        with pytest.raises(OneroConfigError, match="arm positioning is required"):
            DualArm.create()

    def test_arm_spacing_accepted(self):
        try:
            from onero_sdk import DualArm
            from onero_sdk.mujoco_backend import MujocoBackend
        except ImportError:
            pytest.skip("MuJoCo not installed")
        left_be = MujocoBackend(model_name="A1L")
        right_be = MujocoBackend(model_name="A1R")
        dual = DualArm.create(arm_spacing=0.30,
                              left_backend=left_be, right_backend=right_be)
        assert dual.left is not None
        assert dual.right is not None

    def test_explicit_positions_accepted(self):
        try:
            from onero_sdk import DualArm
            from onero_sdk.mujoco_backend import MujocoBackend
        except ImportError:
            pytest.skip("MuJoCo not installed")
        left_be = MujocoBackend(model_name="A1L")
        right_be = MujocoBackend(model_name="A1R")
        dual = DualArm.create(
            base_position_left=[0, 0.2, 0],
            base_position_right=[0, -0.2, 0],
            left_backend=left_be, right_backend=right_be,
        )
        assert dual.left is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
