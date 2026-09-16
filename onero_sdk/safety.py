"""
Safety limits for real-time control.

Provides velocity clamping, joint limit enforcement, and communication
timeout detection. Active by default, can be explicitly disabled.
"""

import math
import time
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SafetyConfig:
    """Configuration for safety limits."""
    max_velocity: float = 6.0           # global ceiling; model limits may be lower
    max_delta_per_step: float = 0.03    # rad per control step
    control_freq: float = 200.0         # Hz — control loop frequency
    timeout_frames: int = 10            # consecutive missed frames before emergency stop
    collision_threshold: float = 0.0    # Nm torque residual threshold (0 = disabled)
    collision_stop: bool = True         # whether collision triggers emergency stop
    # Servo tracking-error watchdog. Two-tier profile mirroring industrial
    # controllers (UR / Franka / ABB): a hard threshold that fires fast for
    # catastrophic deviation (external pull, mechanical jam, encoder failure),
    # and a soft threshold that must be sustained to fire (rules out normal
    # lag-peaks during fast motion).
    #
    # A1 tuning notes (200 Hz control, MIT-protocol servo):
    #  - Measured mid-motion lag on movej at v_max=2 rad/s peaks at ~0.3–0.35 rad.
    #  - Soft threshold 0.5 rad leaves 40–60% headroom over normal lag.
    #  - Hard threshold 1.0 rad is unreachable by normal motion in 50 ms
    #    (v_max × 50 ms = 0.1 rad).
    #  - Soft window 500 ms ≈ human reaction time; long enough to bridge the
    #    lag peak of a large planned motion, short enough to catch faults
    #    before further damage.
    #  - Hard window 50 ms = 10 frames — fast enough to matter.
    #
    # Set the corresponding threshold to 0 to disable either tier.
    tracking_error_hard_threshold: float = 1.0   # rad — immediate fault
    tracking_error_hard_frames: int = 10         # frames (@200 Hz → 50 ms)
    tracking_error_soft_threshold: float = 0.5   # rad — sustained fault
    tracking_error_soft_frames: int = 100        # frames (@200 Hz → 500 ms)
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
                 on_stop: Optional[Callable] = None,
                 joint_velocity_limits: Optional[list[float]] = None):
        self._num_joints = num_joints
        self._joint_limits = joint_limits  # [(p_min, p_max), ...] per joint
        self._config = config or SafetyConfig()
        if joint_velocity_limits is None:
            joint_velocity_limits = [self._config.max_velocity] * num_joints
        if len(joint_velocity_limits) != num_joints:
            raise ValueError("joint_velocity_limits length must match num_joints")
        self._joint_velocity_limits = [
            min(float(limit), self._config.max_velocity)
            for limit in joint_velocity_limits
        ]
        self._on_error = on_error
        self._on_stop = on_stop  # callable to immediately disable motors

        self._last_target: Optional[np.ndarray] = None
        self._miss_counts = [0] * num_joints
        self._tracking_error_hard_count = 0
        self._tracking_error_soft_count = 0
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
    def max_rate(self) -> float:
        """Maximum velocity (rad/s) consistent with per-step delta limit at control freq."""
        return self._config.max_delta_per_step * self._config.control_freq

    @property
    def max_velocity(self) -> float:
        """Largest effective joint velocity ceiling in rad/s."""
        return max(self._joint_velocity_limits, default=self._config.max_velocity)

    @property
    def joint_velocity_limits(self) -> tuple[float, ...]:
        """Effective per-joint velocity ceilings in rad/s."""
        return tuple(self._joint_velocity_limits)

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

        # Replace NaN/Inf with last known good target (or 0)
        bad_mask = ~np.isfinite(result)
        if np.any(bad_mask):
            if self._last_target is not None:
                result[bad_mask] = self._last_target[bad_mask]
            else:
                result[bad_mask] = 0.0

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

    def clamp_joint_limits(self, target: list[float]) -> list[float]:
        """Clamp positions to joint limits only (no rate limiting)."""
        if not self._config.enabled:
            return target
        result = np.array(target, dtype=float)
        # Replace NaN/Inf with last known good target (or 0)
        bad_mask = ~np.isfinite(result)
        if np.any(bad_mask):
            if self._last_target is not None:
                result[bad_mask] = self._last_target[bad_mask]
            else:
                result[bad_mask] = 0.0
        for i, (p_min, p_max) in enumerate(self._joint_limits):
            result[i] = np.clip(result[i], p_min, p_max)
        return result.tolist()

    def reconcile_position(self, measured: list[float]) -> None:
        """
        Force the rate-limiter baseline to match `measured`.

        Only call at explicit event boundaries — never every control frame:
        - after enable()
        - after clearing emergency stop
        - when transitioning out of freedrive / passive states

        Continuous per-frame reconcile is a bug: it turns the rate limiter
        into "target may not lead measured by more than max_delta_per_step",
        which forms a positive feedback loop with servo lag and can drag the
        commanded target back toward measured (observed on movej: joint drifts
        away from home instead of converging). See docs/servo_reconcile_note.
        Use `check_tracking_error` for continuous fault detection instead.
        """
        if self._last_target is None:
            return
        measured_arr = np.array(measured, dtype=float)
        # Reject NaN/Inf — keep previous baseline for corrupted joints
        bad_mask = ~np.isfinite(measured_arr)
        if np.any(bad_mask):
            measured_arr[bad_mask] = self._last_target[bad_mask]
        self._last_target = measured_arr
        self._tracking_error_hard_count = 0
        self._tracking_error_soft_count = 0

    def check_tracking_error(self, measured: list[float],
                             commanded: list[float]) -> None:
        """
        Two-tier watchdog on |measured - commanded|.

        - Hard tier: fires within `tracking_error_hard_frames` frames of any
          sample exceeding `tracking_error_hard_threshold`. For catastrophic
          deviation (external pull, jam, encoder failure).
        - Soft tier: must exceed `tracking_error_soft_threshold` for
          `tracking_error_soft_frames` consecutive frames. Rules out normal
          mid-motion lag peaks.

        Does not mutate the rate-limiter baseline — commanded target remains
        authoritative. Servo lag within the soft threshold is normal and
        ignored; large or sustained deviation signals a fault, not a case
        for silent clamping.
        """
        if not self._config.enabled:
            return

        measured_arr = np.asarray(measured, dtype=float)
        commanded_arr = np.asarray(commanded, dtype=float)
        if measured_arr.shape != commanded_arr.shape:
            return
        finite = np.isfinite(measured_arr) & np.isfinite(commanded_arr)
        if not np.any(finite):
            return

        diff = np.abs(measured_arr - commanded_arr)
        diff[~finite] = 0.0
        max_err = float(np.max(diff))

        hard_th = self._config.tracking_error_hard_threshold
        if hard_th > 0:
            if max_err > hard_th:
                self._tracking_error_hard_count += 1
                if self._tracking_error_hard_count >= self._config.tracking_error_hard_frames:
                    joint_idx = int(np.argmax(diff))
                    self._trigger_emergency_stop(
                        joint_idx,
                        f"Tracking error HARD watchdog: |measured - commanded| = "
                        f"{max_err:.3f} rad on joint {joint_idx} for "
                        f"{self._tracking_error_hard_count} consecutive frames "
                        f"(threshold {hard_th} rad, "
                        f"{self._config.tracking_error_hard_frames} frames)"
                    )
                    return
            else:
                self._tracking_error_hard_count = 0

        soft_th = self._config.tracking_error_soft_threshold
        if soft_th > 0:
            if max_err > soft_th:
                self._tracking_error_soft_count += 1
                if self._tracking_error_soft_count >= self._config.tracking_error_soft_frames:
                    joint_idx = int(np.argmax(diff))
                    self._trigger_emergency_stop(
                        joint_idx,
                        f"Tracking error SOFT watchdog: |measured - commanded| = "
                        f"{max_err:.3f} rad on joint {joint_idx} for "
                        f"{self._tracking_error_soft_count} consecutive frames "
                        f"(threshold {soft_th} rad, "
                        f"{self._config.tracking_error_soft_frames} frames)"
                    )
            else:
                self._tracking_error_soft_count = 0

    def clamp_velocity(self, velocities: list[float],
                       positions: Optional[list[float]] = None) -> list[float]:
        """Safely scale joint velocities, preserving their relative direction.

        The vector is scaled as a whole to satisfy the global velocity ceiling.
        When measured positions are supplied, it is scaled again so the next
        control step cannot cross any configured joint limit.
        """
        if not self._config.enabled:
            return velocities
        finite = [v if math.isfinite(v) else 0.0 for v in velocities]
        scale = 1.0
        for velocity, limit in zip(finite, self._joint_velocity_limits):
            if velocity:
                scale = min(scale, limit / abs(velocity))

        if positions is not None:
            dt = 1.0 / self._config.control_freq
            for q, v, (lower, upper) in zip(positions, finite, self._joint_limits):
                scaled_v = v * scale
                if scaled_v > 0:
                    room = max(0.0, upper - q)
                    scale = min(scale, room / (v * dt))
                elif scaled_v < 0:
                    room = max(0.0, q - lower)
                    scale = min(scale, room / (-v * dt))

        scale = max(0.0, min(1.0, scale))
        return [v * scale for v in finite]

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
        self._tracking_error_hard_count = 0
        self._tracking_error_soft_count = 0
        if initial_positions is not None:
            self._last_target = np.array(initial_positions, dtype=float)
        else:
            self._last_target = None

    def set_emergency_stop(self) -> None:
        """Set the motion-abort flag. Running motions will exit their control loop.

        This only sets an internal flag — it does NOT disable the backend hardware.
        For a full hardware stop (disable motors + notify), use trigger_emergency_stop().
        """
        self._emergency_stop = True

    def clear_emergency_stop(self) -> None:
        """Clear emergency stop flag without resetting other state."""
        self._emergency_stop = False

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

    def trigger_emergency_stop(self, joint_index: int, reason: str) -> None:
        """Trigger emergency stop — disables arm and notifies user.

        Public API for use by subsystems (e.g. collision detection).
        """
        self._trigger_emergency_stop(joint_index, reason)

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
