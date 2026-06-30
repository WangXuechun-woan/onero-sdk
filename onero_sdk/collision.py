"""
Multi-strategy collision detection — works without accurate payload model.

Three independent channels detect collisions through different physical signals:
- Impact (dτ/dt): catches fast collisions via torque rate-of-change spike
- Energy (P=τ·ω): catches medium-speed collisions via power anomaly
- Residual (model-based): catches slow pushes (optional, needs dynamics model)

The adaptive baseline in Impact/Energy channels naturally tracks slow payload
changes (e.g., lifting a garment) while still detecting sudden collision spikes.
"""

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


class CollisionResponse(Enum):
    """What to do when collision is detected."""
    STOP = "stop"
    CALLBACK = "callback"
    BOTH = "both"


@dataclass
class CollisionConfig:
    """Configuration for multi-strategy collision detection.

    Default parameters work for most scenarios without any tuning.
    All thresholds are per-joint — fires if ANY joint exceeds on any enabled channel.
    """

    # Impact channel (primary — no model needed)
    impact_enabled: bool = True
    impact_threshold: float = 50.0
    impact_baseline_alpha: float = 0.02
    impact_noise_floor: float = 5.0
    impact_holdoff_frames: int = 10

    # Energy channel (secondary — no model needed)
    energy_enabled: bool = True
    energy_threshold: float = 15.0
    energy_baseline_alpha: float = 0.05
    energy_holdoff_frames: int = 10

    # Residual channel (optional — needs dynamics model)
    residual_enabled: bool = False
    residual_threshold: float = 8.0

    # General
    response: CollisionResponse = CollisionResponse.STOP
    min_velocity: float = 0.05
    warmup_frames: int = 20
    control_freq: float = 200.0

    def __post_init__(self):
        if not isinstance(self.response, CollisionResponse):
            raise TypeError(
                f"response must be a CollisionResponse enum, got {type(self.response).__name__}: {self.response!r}"
            )
        if not (0.0 <= self.impact_baseline_alpha <= 1.0):
            raise ValueError(f"impact_baseline_alpha must be in [0, 1], got {self.impact_baseline_alpha}")
        if not (0.0 <= self.energy_baseline_alpha <= 1.0):
            raise ValueError(f"energy_baseline_alpha must be in [0, 1], got {self.energy_baseline_alpha}")
        if self.impact_threshold < 0 or not math.isfinite(self.impact_threshold):
            raise ValueError(f"impact_threshold must be finite and non-negative, got {self.impact_threshold}")
        if self.energy_threshold < 0 or not math.isfinite(self.energy_threshold):
            raise ValueError(f"energy_threshold must be finite and non-negative, got {self.energy_threshold}")
        if self.residual_threshold < 0 or not math.isfinite(self.residual_threshold):
            raise ValueError(f"residual_threshold must be finite and non-negative, got {self.residual_threshold}")
        if self.control_freq <= 0 or not math.isfinite(self.control_freq):
            raise ValueError(f"control_freq must be finite and positive, got {self.control_freq}")
        if self.warmup_frames < 0:
            raise ValueError(f"warmup_frames must be non-negative, got {self.warmup_frames}")
        if self.impact_holdoff_frames < 0:
            raise ValueError(f"impact_holdoff_frames must be non-negative, got {self.impact_holdoff_frames}")
        if self.energy_holdoff_frames < 0:
            raise ValueError(f"energy_holdoff_frames must be non-negative, got {self.energy_holdoff_frames}")
        if self.min_velocity < 0 or not math.isfinite(self.min_velocity):
            raise ValueError(f"min_velocity must be finite and non-negative, got {self.min_velocity}")
        if self.impact_noise_floor < 0 or not math.isfinite(self.impact_noise_floor):
            raise ValueError(f"impact_noise_floor must be finite and non-negative, got {self.impact_noise_floor}")


@dataclass
class CollisionEvent:
    """Information about a detected collision."""
    timestamp: float
    channel: str
    joint_index: int
    magnitude: float
    threshold: float
    torques: list[float] = field(default_factory=list)
    velocities: list[float] = field(default_factory=list)
    joints: list[float] = field(default_factory=list)

    @property
    def severity(self) -> float:
        """Ratio of magnitude to threshold (>1.0 means triggered)."""
        return self.magnitude / self.threshold if self.threshold > 0 else 0.0


class CollisionDetector:
    """
    Multi-strategy collision detector for real-time control loops.

    Called once per control frame with fresh state data. Uses numpy-vectorized
    operations for minimal overhead (~30μs per frame at 7 joints).
    """

    def __init__(
        self,
        num_joints: int,
        config: CollisionConfig,
        on_collision: Optional[Callable[["CollisionEvent"], None]] = None,
        residual_fn: Optional[Callable[[list, list, list], list]] = None,
    ):
        if num_joints <= 0:
            raise ValueError(f"num_joints must be positive, got {num_joints}")

        self._n = num_joints
        self._config = config
        self._on_collision = on_collision
        self._residual_fn = residual_fn
        self._dt = 1.0 / config.control_freq

        # Impact channel state
        self._prev_torques: Optional[np.ndarray] = None
        self._dtau_baseline = np.zeros(num_joints)
        self._impact_holdoff = np.zeros(num_joints, dtype=int)

        # Energy channel state
        self._power_baseline = np.zeros(num_joints)
        self._energy_holdoff = np.zeros(num_joints, dtype=int)

        # General state
        self._frame_count = 0
        self._armed = False
        self._last_event: Optional[CollisionEvent] = None
        self._residual_error_count = 0

    @property
    def armed(self) -> bool:
        return self._armed

    @property
    def last_event(self) -> Optional[CollisionEvent]:
        return self._last_event

    @property
    def response(self) -> CollisionResponse:
        return self._config.response

    def reset(self) -> None:
        """Reset all internal state. Call at motion start."""
        self._prev_torques = None
        self._dtau_baseline[:] = 0.0
        self._impact_holdoff[:] = 0
        self._power_baseline[:] = 0.0
        self._energy_holdoff[:] = 0
        self._frame_count = 0
        self._armed = False
        self._last_event = None
        self._residual_error_count = 0

    def process_frame(
        self,
        joints: list[float],
        velocities: list[float],
        torques: list[float],
    ) -> Optional[CollisionEvent]:
        """
        Process one control frame. Returns CollisionEvent if detected, else None.

        Must be called at control_freq rate. Note: the Impact channel requires
        two consecutive frames to compute dτ/dt, so it cannot fire on the very
        first frame even with warmup_frames=0. Energy and Residual channels
        have no such constraint.
        """
        tau = np.asarray(torques, dtype=np.float64)
        vel = np.asarray(velocities, dtype=np.float64)
        pos = np.asarray(joints, dtype=np.float64)

        # Guard against wrong shape or NaN/Inf
        if tau.shape != (self._n,) or vel.shape != (self._n,) or pos.shape != (self._n,):
            return None
        if not (np.all(np.isfinite(tau)) and np.all(np.isfinite(vel))
                and np.all(np.isfinite(pos))):
            return None

        self._frame_count += 1

        # Decrement holdoff counters
        self._impact_holdoff = np.maximum(0, self._impact_holdoff - 1)
        self._energy_holdoff = np.maximum(0, self._energy_holdoff - 1)

        # Skip impact/energy if arm nearly static (noise-dominated regime)
        # Residual channel is exempt — it's specifically for slow/static contact
        max_vel = float(np.max(np.abs(vel)))
        moving = max_vel >= self._config.min_velocity

        event = None

        # Impact channel (requires motion)
        if moving and self._config.impact_enabled and self._prev_torques is not None:
            event = self._check_impact(tau, vel, pos)

        # Energy channel (requires motion)
        if event is None and moving and self._config.energy_enabled:
            event = self._check_energy(tau, vel, pos)

        # Residual channel (works at any velocity)
        if (
            event is None
            and self._config.residual_enabled
            and self._residual_fn is not None
        ):
            event = self._check_residual(pos, vel, tau)

        # Update previous state
        self._prev_torques = tau.copy()

        # Warmup guard — never trigger during baseline fill
        if not self._armed:
            if self._frame_count >= self._config.warmup_frames:
                self._armed = True
            else:
                return None

        if event is not None:
            self._last_event = event
            if event.channel == "impact":
                self._impact_holdoff[:] = self._config.impact_holdoff_frames
            elif event.channel == "energy":
                self._energy_holdoff[:] = self._config.energy_holdoff_frames
            if self._on_collision is not None:
                try:
                    self._on_collision(event)
                except Exception:
                    pass

        return event

    def _check_impact(
        self, tau: np.ndarray, vel: np.ndarray, pos: np.ndarray
    ) -> Optional[CollisionEvent]:
        dtau_dt = np.abs(tau - self._prev_torques) / self._dt

        alpha = self._config.impact_baseline_alpha
        self._dtau_baseline = alpha * dtau_dt + (1.0 - alpha) * self._dtau_baseline

        spike = dtau_dt - self._dtau_baseline
        threshold = max(self._config.impact_threshold, self._config.impact_noise_floor)

        # Mask joints in holdoff
        candidates = spike * (self._impact_holdoff == 0).astype(np.float64)

        max_idx = int(np.argmax(candidates))
        max_spike = float(candidates[max_idx])

        if max_spike > threshold:
            return CollisionEvent(
                timestamp=time.monotonic(),
                channel="impact",
                joint_index=max_idx,
                magnitude=max_spike,
                threshold=threshold,
                torques=tau.tolist(),
                velocities=vel.tolist(),
                joints=pos.tolist(),
            )
        return None

    def _check_energy(
        self, tau: np.ndarray, vel: np.ndarray, pos: np.ndarray
    ) -> Optional[CollisionEvent]:
        power = np.abs(tau * vel)

        alpha = self._config.energy_baseline_alpha
        self._power_baseline = alpha * power + (1.0 - alpha) * self._power_baseline

        anomaly = power - self._power_baseline

        # Mask joints in holdoff
        candidates = anomaly * (self._energy_holdoff == 0).astype(np.float64)

        max_idx = int(np.argmax(candidates))
        max_anomaly = float(candidates[max_idx])

        if max_anomaly > self._config.energy_threshold:
            return CollisionEvent(
                timestamp=time.monotonic(),
                channel="energy",
                joint_index=max_idx,
                magnitude=max_anomaly,
                threshold=self._config.energy_threshold,
                torques=tau.tolist(),
                velocities=vel.tolist(),
                joints=pos.tolist(),
            )
        return None

    def _check_residual(
        self, pos: np.ndarray, vel: np.ndarray, tau: np.ndarray
    ) -> Optional[CollisionEvent]:
        try:
            residuals = np.asarray(
                self._residual_fn(pos.tolist(), vel.tolist(), tau.tolist()),
                dtype=np.float64,
            )
        except Exception as e:
            self._warn_residual_error(f"residual_fn raised {type(e).__name__}: {e}")
            return None
        if residuals.shape != (self._n,):
            self._warn_residual_error(
                f"residual_fn returned shape {residuals.shape}, expected ({self._n},)")
            return None
        if not np.all(np.isfinite(residuals)):
            self._warn_residual_error("residual_fn returned NaN/Inf values")
            return None
        max_idx = int(np.argmax(np.abs(residuals)))
        max_residual = float(abs(residuals[max_idx]))

        if max_residual > self._config.residual_threshold:
            return CollisionEvent(
                timestamp=time.monotonic(),
                channel="residual",
                joint_index=max_idx,
                magnitude=max_residual,
                threshold=self._config.residual_threshold,
                torques=tau.tolist(),
                velocities=vel.tolist(),
                joints=pos.tolist(),
            )
        return None

    def _warn_residual_error(self, msg: str) -> None:
        self._residual_error_count += 1
        if self._residual_error_count == 1 or self._residual_error_count % 200 == 0:
            logger.warning(
                "Residual collision channel degraded (%d errors): %s",
                self._residual_error_count, msg,
            )
