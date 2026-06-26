"""
Safety limits for real-time control.

Provides velocity clamping, joint limit enforcement, and communication
timeout detection. Active by default, can be explicitly disabled.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SafetyConfig:
    """Configuration for safety limits."""
    max_velocity: float = 5.0           # rad/s per joint
    max_delta_per_step: float = 0.03    # rad per control step
    timeout_frames: int = 10            # consecutive missed frames before emergency stop
    collision_threshold: float = 0.0    # Nm torque residual threshold (0 = disabled)
    collision_stop: bool = True         # whether collision triggers emergency stop
    enabled: bool = True


class SafetyMonitor:
    """
    Monitors arm safety in real-time control mode.

    - Clamps velocity/position changes per step
    - Enforces joint position limits
    - Detects communication timeouts
    - Triggers emergency stop on critical faults
    """

    def __init__(self, num_joints: int, joint_limits: list[tuple[float, float]],
                 config: SafetyConfig = None,
                 on_error: Optional[Callable] = None,
                 on_stop: Optional[Callable] = None):
        self._num_joints = num_joints
        self._joint_limits = joint_limits  # [(p_min, p_max), ...] per joint
        self._config = config or SafetyConfig()
        self._on_error = on_error
        self._on_stop = on_stop  # callable to immediately disable motors

        self._last_target: Optional[np.ndarray] = None
        self._miss_counts = [0] * num_joints
        self._emergency_stop = False

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._config.enabled = value

    @property
    def emergency_stopped(self) -> bool:
        return self._emergency_stop

    @property
    def on_error(self) -> Optional[Callable]:
        return self._on_error

    @on_error.setter
    def on_error(self, handler: Optional[Callable]):
        self._on_error = handler

    def clamp_target(self, target: list[float]) -> list[float]:
        """Apply safety limits to a servo target. Returns clamped target."""
        if not self._config.enabled:
            return target

        result = np.array(target, dtype=float)

        # Joint position limits
        for i, (p_min, p_max) in enumerate(self._joint_limits):
            result[i] = np.clip(result[i], p_min, p_max)

        # Rate limiting (max change per step)
        if self._last_target is not None:
            delta = result - self._last_target
            delta_clamped = np.clip(
                delta,
                -self._config.max_delta_per_step,
                self._config.max_delta_per_step
            )
            result = self._last_target + delta_clamped

        self._last_target = result
        return result.tolist()

    def reconcile_position(self, measured: list[float]) -> None:
        """
        Reconcile rate limiter baseline with actual measured position.

        Call periodically (e.g., each control frame) to prevent the internal
        baseline from diverging too far from reality when the arm cannot
        track commanded targets (contact, payload, controller lag).
        """
        if self._last_target is None:
            return
        measured_arr = np.array(measured, dtype=float)
        self._last_target = measured_arr

    def clamp_velocity(self, velocities: list[float]) -> list[float]:
        """Clamp joint velocities to max_velocity. Returns clamped velocities."""
        if not self._config.enabled:
            return velocities
        limit = self._config.max_velocity
        return [max(-limit, min(limit, v)) for v in velocities]

    def record_response(self, joint_index: int, received: bool) -> None:
        """Record whether a joint responded this cycle."""
        if not self._config.enabled:
            return

        if received:
            self._miss_counts[joint_index] = 0
        else:
            self._miss_counts[joint_index] += 1
            if self._miss_counts[joint_index] >= self._config.timeout_frames:
                self._trigger_emergency_stop(
                    joint_index,
                    f"Communication timeout: {self._miss_counts[joint_index]} "
                    f"consecutive frames missed on joint {joint_index}"
                )

    def record_hardware_fault(self, joint_index: int, error_code: int) -> None:
        """Record a hardware fault from motor feedback."""
        if not self._config.enabled:
            return

        from .protocol import MotorError
        critical_errors = {
            MotorError.OVERVOLTAGE, MotorError.OVERCURRENT,
            MotorError.MOSFET_OVERTEMP, MotorError.COIL_OVERTEMP,
            MotorError.OVERLOAD,
        }

        if error_code in critical_errors:
            try:
                desc = MotorError(error_code).name
            except ValueError:
                desc = f"0x{error_code:X}"
            self._trigger_emergency_stop(
                joint_index,
                f"Critical hardware fault on joint {joint_index}: {desc}"
            )
        elif error_code == MotorError.COMM_LOSS:
            logger.warning(f"Communication loss reported by joint {joint_index}")

    def reset(self, initial_positions: Optional[list[float]] = None) -> None:
        """Reset safety monitor state (e.g., after re-enable)."""
        self._emergency_stop = False
        self._miss_counts = [0] * self._num_joints
        if initial_positions is not None:
            self._last_target = np.array(initial_positions, dtype=float)
        else:
            self._last_target = None

    def set_emergency_stop(self) -> None:
        """Programmatically trigger emergency stop from application code."""
        self._emergency_stop = True

    def check_collision(self, torque_residuals: list[float]) -> bool:
        """
        Check for collision based on torque residuals.

        Args:
            torque_residuals: Per-joint torque residual (measured - expected).

        Returns:
            True if collision detected.
        """
        if not self._config.enabled:
            return False
        if self._config.collision_threshold <= 0:
            return False

        max_residual = max(abs(r) for r in torque_residuals)
        if max_residual > self._config.collision_threshold:
            joint_idx = int(np.argmax(np.abs(torque_residuals)))
            logger.warning(
                f"Collision detected: joint {joint_idx} residual "
                f"{torque_residuals[joint_idx]:.2f} Nm (threshold: {self._config.collision_threshold})"
            )
            if self._config.collision_stop:
                self._trigger_emergency_stop(
                    joint_idx,
                    f"Collision detected: residual {max_residual:.2f} Nm > "
                    f"threshold {self._config.collision_threshold} Nm"
                )
            return True
        return False

    def _trigger_emergency_stop(self, joint_index: int, reason: str) -> None:
        """Trigger emergency stop — disables arm and notifies user."""
        if self._emergency_stop:
            return  # already stopped

        self._emergency_stop = True
        logger.error(f"EMERGENCY STOP: {reason}")

        # Immediately disable motors
        if self._on_stop:
            try:
                self._on_stop()
            except Exception:
                pass

        if self._on_error:
            from .exceptions import OneroHardwareError
            error = OneroHardwareError(
                joint_index=joint_index,
                error_code=0,
                description=reason,
            )
            try:
                self._on_error(error)
            except Exception:
                pass
