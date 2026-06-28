"""
SafeBackend & SafetyMonitor algorithm tests — rate limiting, velocity capping.

Tests use equivalence class partitioning and boundary value analysis:
- Within-limit (passthrough): commands that should NOT be modified
- At-boundary: commands exactly at the limit edge
- Over-limit: commands that MUST be clamped
- Oscillation prevention: repeated clamping must converge, not oscillate
- command_hold bypass: force_mode/freedrive path must skip rate limiting

No simulation — uses mock backend to isolate the safety algorithm.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from onero_sdk.backend import ArmBackend, ArmState
from onero_sdk.safety import SafetyMonitor, SafetyConfig
from onero_sdk.safe_backend import SafeBackend


class MockInnerBackend(ArmBackend):
    """Minimal mock backend that records commands."""

    def __init__(self, num_joints=7):
        self._num_joints = num_joints
        self.last_command = None
        self._state = ArmState(
            joints=[0.0] * num_joints,
            velocities=[0.0] * num_joints,
            torques=[0.0] * num_joints,
            timestamp=0.0,
        )

    @property
    def num_joints(self) -> int:
        return self._num_joints

    def enable(self) -> ArmState:
        return self._state

    def disable(self) -> None:
        pass

    def command(self, positions, velocities, kps, kds, torques) -> ArmState:
        self.last_command = {
            "positions": list(positions),
            "velocities": list(velocities),
            "kps": list(kps),
            "kds": list(kds),
            "torques": list(torques),
        }
        self._state = ArmState(
            joints=list(positions),
            velocities=list(velocities),
            torques=list(torques),
            timestamp=self._state.timestamp + 0.005,
        )
        return self._state

    def command_velocity(self, velocities) -> ArmState:
        return self._state

    def command_torque(self, torques) -> ArmState:
        return self._state

    def get_state(self) -> ArmState:
        return self._state

    def close(self) -> None:
        pass


def _make_safe_backend(max_delta=0.03, max_velocity=5.0, torque_limits=None):
    """Create SafeBackend with configurable safety parameters."""
    inner = MockInnerBackend()
    joint_limits = [(-3.14, 3.14)] * 7
    config = SafetyConfig(
        max_delta_per_step=max_delta,
        max_velocity=max_velocity,
        control_freq=200.0,
    )
    safety = SafetyMonitor(
        num_joints=7,
        joint_limits=joint_limits,
        config=config,
    )
    # Initialize baseline at zero
    safety.reset(initial_positions=[0.0] * 7)
    sb = SafeBackend(inner=inner, safety=safety, torque_limits=torque_limits)
    return sb, inner, safety


class TestRateLimiting:
    """Rate limiting algorithm correctness."""

    def test_rate_limit_passthrough_within_bounds(self):
        """Commands within max_delta_per_step pass through unchanged."""
        sb, inner, _ = _make_safe_backend(max_delta=0.03)

        # Step of 0.01 rad — well within 0.03 limit
        small_target = [0.01] * 7
        zeros = [0.0] * 7
        kps = [100.0] * 7
        kds = [5.0] * 7

        sb.command(small_target, zeros, kps, kds, zeros)

        # Position should pass through unmodified
        for j in range(7):
            assert abs(inner.last_command["positions"][j] - small_target[j]) < 1e-10, (
                f"Joint {j}: within-limit target was modified"
            )

    def test_rate_limit_boundary_exact_at_limit(self):
        """Command exactly at max_delta_per_step passes through."""
        max_delta = 0.03
        sb, inner, _ = _make_safe_backend(max_delta=max_delta)

        # Exactly at the boundary
        boundary_target = [max_delta] * 7
        zeros = [0.0] * 7
        kps = [100.0] * 7
        kds = [5.0] * 7

        sb.command(boundary_target, zeros, kps, kds, zeros)

        for j in range(7):
            assert abs(inner.last_command["positions"][j] - max_delta) < 1e-10, (
                f"Joint {j}: boundary target was incorrectly clamped"
            )

    def test_rate_limit_caps_velocity_not_zeros(self):
        """
        Over-limit: position is clamped BUT velocity is NOT zeroed.

        Bug regression: old code zeroed velocity when position was clamped,
        causing kd term to brake and oscillate. Correct: cap velocity magnitude.
        """
        max_delta = 0.03
        sb, inner, _ = _make_safe_backend(max_delta=max_delta)

        # Target far beyond one-step limit
        far_target = [1.0] * 7
        high_vel = [2.0] * 7
        kps = [100.0] * 7
        kds = [5.0] * 7
        zeros = [0.0] * 7

        sb.command(far_target, high_vel, kps, kds, zeros)

        for j in range(7):
            # Position must be clamped
            assert abs(inner.last_command["positions"][j]) <= max_delta + 1e-10, (
                f"Joint {j}: position not clamped to max_delta"
            )
            # Velocity must NOT be zero — it should be capped, not killed
            assert abs(inner.last_command["velocities"][j]) > 0, (
                f"Joint {j}: velocity incorrectly zeroed on rate-limited step"
            )
            # Velocity must be capped at max_rate
            max_rate = max_delta * 200.0  # 6.0 rad/s
            assert abs(inner.last_command["velocities"][j]) <= max_rate + 1e-10, (
                f"Joint {j}: velocity exceeds max_rate"
            )

    def test_rate_limit_no_oscillation(self):
        """Repeated clamping: position monotonically approaches target."""
        max_delta = 0.03
        sb, inner, _ = _make_safe_backend(max_delta=max_delta)

        # Command the same far target repeatedly
        far_target = [1.0] * 7
        zeros = [0.0] * 7
        kps = [100.0] * 7
        kds = [5.0] * 7

        positions_over_time = []
        for _ in range(20):
            sb.command(far_target, zeros, kps, kds, zeros)
            positions_over_time.append(inner.last_command["positions"][0])

        # Monotonically increasing (for positive target)
        for i in range(1, len(positions_over_time)):
            assert positions_over_time[i] >= positions_over_time[i-1] - 1e-10, (
                f"Step {i}: position decreased (oscillation): "
                f"{positions_over_time[i]:.6f} < {positions_over_time[i-1]:.6f}"
            )

    def test_reconcile_prevents_baseline_windup(self):
        """After reconcile, rate limiter uses actual position as new baseline."""
        max_delta = 0.03
        sb, inner, safety = _make_safe_backend(max_delta=max_delta)

        # Command forward several steps to build up baseline
        zeros = [0.0] * 7
        kps = [100.0] * 7
        kds = [5.0] * 7

        for _ in range(5):
            sb.command([0.5] * 7, zeros, kps, kds, zeros)

        # Now reconcile to a DIFFERENT actual position (as if arm lagged)
        actual_pos = [0.05] * 7
        safety.reconcile_position(actual_pos)

        # Next command should be limited relative to actual_pos, not accumulated baseline
        sb.command([0.5] * 7, zeros, kps, kds, zeros)
        for j in range(7):
            # Should be at most max_delta from actual_pos
            assert inner.last_command["positions"][j] <= actual_pos[j] + max_delta + 1e-10, (
                f"Joint {j}: position not limited from reconciled baseline"
            )


class TestCommandHold:
    """command_hold() bypasses rate limiting but keeps torque/joint limits."""

    def test_command_hold_bypasses_rate_limit(self):
        """command_hold() does NOT apply rate limiting — large jumps allowed."""
        max_delta = 0.03
        sb, inner, _ = _make_safe_backend(max_delta=max_delta, torque_limits=[50.0]*7)

        # Jump far beyond one step — should pass through hold path
        far_target = [1.0] * 7
        zeros = [0.0] * 7
        kps = [100.0] * 7
        kds = [5.0] * 7

        sb.command_hold(far_target, zeros, kps, kds, zeros)

        for j in range(7):
            # Position NOT clamped by rate limiter
            assert abs(inner.last_command["positions"][j] - 1.0) < 1e-10, (
                f"Joint {j}: command_hold incorrectly rate-limited"
            )

    def test_command_hold_still_clamps_joint_limits(self):
        """command_hold() respects joint position limits."""
        sb, inner, _ = _make_safe_backend(max_delta=0.03)

        # Target beyond joint limits (±3.14)
        over_limit = [5.0] * 7
        zeros = [0.0] * 7
        kps = [100.0] * 7
        kds = [5.0] * 7

        sb.command_hold(over_limit, zeros, kps, kds, zeros)

        for j in range(7):
            assert inner.last_command["positions"][j] <= 3.14, (
                f"Joint {j}: command_hold did not clamp to joint limits"
            )


class TestTorqueClamping:
    """Torque limit clamping behavior."""

    def test_torque_limits_symmetric(self):
        """Positive and negative torques are clamped symmetrically."""
        torque_limits = [10.0, 10.0, 8.0, 8.0, 5.0, 5.0, 3.0]
        sb, inner, _ = _make_safe_backend(torque_limits=torque_limits)

        # Positive over-limit
        high_pos = [100.0] * 7
        zeros = [0.0] * 7
        kps = [100.0] * 7
        kds = [5.0] * 7

        sb.command(zeros, zeros, kps, kds, high_pos)
        for j in range(7):
            assert inner.last_command["torques"][j] <= torque_limits[j] + 1e-10

        # Negative over-limit
        high_neg = [-100.0] * 7
        sb.command(zeros, zeros, kps, kds, high_neg)
        for j in range(7):
            assert inner.last_command["torques"][j] >= -torque_limits[j] - 1e-10

    def test_torque_within_limit_unchanged(self):
        """Torques within limits pass through unchanged."""
        torque_limits = [10.0] * 7
        sb, inner, _ = _make_safe_backend(torque_limits=torque_limits)

        small_torque = [3.0, -2.0, 1.5, -4.0, 2.5, -1.0, 0.5]
        zeros = [0.0] * 7
        kps = [100.0] * 7
        kds = [5.0] * 7

        sb.command(zeros, zeros, kps, kds, small_torque)
        for j in range(7):
            assert abs(inner.last_command["torques"][j] - small_torque[j]) < 1e-10


class TestVelocityClamping:
    """Velocity clamping via command_velocity()."""

    def test_velocity_clamped_at_max(self):
        """Velocities exceeding max_velocity are clamped."""
        max_vel = 5.0
        sb, inner, _ = _make_safe_backend(max_velocity=max_vel)

        high_vel = [10.0, -10.0, 5.0, -5.0, 3.0, -3.0, 7.0]
        sb.command_velocity(high_vel)

        # The inner backend receives the state after velocity command;
        # safety clamps velocity before forwarding
        # Verify via the SafetyMonitor's clamp behavior directly
        from onero_sdk.safety import SafetyMonitor, SafetyConfig
        config = SafetyConfig(max_velocity=max_vel)
        monitor = SafetyMonitor(num_joints=7, joint_limits=[(-3.14, 3.14)]*7, config=config)
        clamped = monitor.clamp_velocity(high_vel)
        for j in range(7):
            assert abs(clamped[j]) <= max_vel + 1e-10, (
                f"Joint {j}: velocity {clamped[j]} exceeds max {max_vel}"
            )


class TestBoundaryValues:
    """Edge cases and boundary values for safety system."""

    def test_zero_movement_passthrough(self):
        """Zero command (no movement) passes through unchanged."""
        sb, inner, _ = _make_safe_backend()
        zeros = [0.0] * 7
        kps = [100.0] * 7
        kds = [5.0] * 7

        sb.command(zeros, zeros, kps, kds, zeros)
        for j in range(7):
            assert inner.last_command["positions"][j] == 0.0
            assert inner.last_command["velocities"][j] == 0.0

    def test_negative_direction_rate_limit(self):
        """Rate limiting works symmetrically for negative direction."""
        max_delta = 0.03
        sb, inner, _ = _make_safe_backend(max_delta=max_delta)

        # Start from zero, go negative
        neg_target = [-1.0] * 7
        zeros = [0.0] * 7
        kps = [100.0] * 7
        kds = [5.0] * 7

        sb.command(neg_target, zeros, kps, kds, zeros)

        for j in range(7):
            assert inner.last_command["positions"][j] >= -max_delta - 1e-10, (
                f"Joint {j}: negative direction not clamped correctly"
            )

    def test_single_joint_over_limit(self):
        """Only the over-limit joint is clamped, others pass through."""
        max_delta = 0.03
        sb, inner, _ = _make_safe_backend(max_delta=max_delta)

        # Joint 0 far, others within limit
        target = [1.0, 0.01, 0.02, 0.01, 0.015, 0.005, 0.0]
        zeros = [0.0] * 7
        kps = [100.0] * 7
        kds = [5.0] * 7

        sb.command(target, zeros, kps, kds, zeros)

        # Joint 0 should be clamped
        assert abs(inner.last_command["positions"][0]) <= max_delta + 1e-10
        # Other joints should pass through unchanged
        for j in range(1, 7):
            assert abs(inner.last_command["positions"][j] - target[j]) < 1e-10, (
                f"Joint {j}: incorrectly clamped (was within limit)"
            )


class TestEmergencyStop:
    """Emergency stop blocks further commands."""

    def test_emergency_stop_blocks_command(self):
        """After emergency stop, command() raises."""
        sb, inner, safety = _make_safe_backend()
        zeros = [0.0] * 7
        kps = [100.0] * 7
        kds = [5.0] * 7

        safety.set_emergency_stop()

        from onero_sdk.exceptions import OneroHardwareError
        with pytest.raises(OneroHardwareError):
            sb.command(zeros, zeros, kps, kds, zeros)

    def test_emergency_stop_blocks_velocity(self):
        """After emergency stop, command_velocity() raises."""
        sb, inner, safety = _make_safe_backend()
        safety.set_emergency_stop()

        from onero_sdk.exceptions import OneroHardwareError
        with pytest.raises(OneroHardwareError):
            sb.command_velocity([0.0] * 7)

    def test_clear_emergency_allows_commands(self):
        """After clearing emergency stop, commands work again."""
        sb, inner, safety = _make_safe_backend()
        zeros = [0.0] * 7
        kps = [100.0] * 7
        kds = [5.0] * 7

        safety.set_emergency_stop()
        safety.clear_emergency_stop()

        # Should not raise
        sb.command(zeros, zeros, kps, kds, zeros)


class TestCollisionDetection:
    """SafetyMonitor collision detection."""

    def test_collision_below_threshold_no_trigger(self):
        """Residuals below threshold do not trigger collision."""
        joint_limits = [(-3.14, 3.14)] * 7
        config = SafetyConfig(collision_threshold=5.0, collision_stop=True)
        safety = SafetyMonitor(num_joints=7, joint_limits=joint_limits, config=config)

        result = safety.check_collision([1.0, 2.0, 1.5, 0.5, 1.0, 0.8, 0.3])
        assert result is False
        assert not safety.emergency_stopped

    def test_collision_above_threshold_triggers(self):
        """Residuals above threshold trigger collision and emergency stop."""
        joint_limits = [(-3.14, 3.14)] * 7
        config = SafetyConfig(collision_threshold=5.0, collision_stop=True)
        safety = SafetyMonitor(num_joints=7, joint_limits=joint_limits, config=config)

        result = safety.check_collision([1.0, 2.0, 10.0, 0.5, 1.0, 0.8, 0.3])
        assert result is True
        assert safety.emergency_stopped

    def test_collision_disabled_when_threshold_zero(self):
        """collision_threshold=0 means collision detection is disabled."""
        joint_limits = [(-3.14, 3.14)] * 7
        config = SafetyConfig(collision_threshold=0.0)
        safety = SafetyMonitor(num_joints=7, joint_limits=joint_limits, config=config)

        result = safety.check_collision([100.0] * 7)
        assert result is False

    def test_collision_no_stop_when_collision_stop_false(self):
        """collision_stop=False: returns True but no emergency stop."""
        joint_limits = [(-3.14, 3.14)] * 7
        config = SafetyConfig(collision_threshold=5.0, collision_stop=False)
        safety = SafetyMonitor(num_joints=7, joint_limits=joint_limits, config=config)

        result = safety.check_collision([10.0] * 7)
        assert result is True
        assert not safety.emergency_stopped


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
