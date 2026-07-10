"""
Adaptive gain policy — compute per-joint kp/kd from mass matrix and desired dynamics.

Uses the relationship between impedance parameters and system dynamics:
  kp_i = M_ii(q) * ωn²
  kd_i = 2 * ζ * sqrt(kp_i * M_ii(q))

This ensures consistent dynamic response across all configurations:
the arm feels equally "stiff" or "soft" regardless of pose.
"""

from enum import Enum
import logging

import numpy as np

logger = logging.getLogger(__name__)


class Stiffness(Enum):
    """Impedance presets defined by natural frequency and damping ratio."""

    STIFF = "stiff"
    MEDIUM = "medium"
    SOFT = "soft"

    @property
    def omega_n(self) -> float:
        return _PARAMS[self.value][0]

    @property
    def zeta(self) -> float:
        return _PARAMS[self.value][1]


_PARAMS: dict[str, tuple[float, float]] = {
    "stiff": (40.0, 1.0),
    "medium": (20.0, 1.0),
    "soft": (8.0, 0.7),
}

# MIT protocol encoding limits
_KP_MAX = 500.0
_KD_MAX = 5.0

# Adaptive-gain safety floor. When URDF inertias underestimate a distal link
# (e.g. A1's Link7 CAD export lists 17.8 g), M_ii → 0 makes M×ωn² collapse to
# near-zero kp, and the servo cannot hold against friction/cogging. Compare
# with industry practice: Franka Panda's smallest joint stiffness (flange
# joint 7) is 50 Nm/rad, UR5's Gazebo controller uses kp=10 on wrist3. Our
# floor sits between those, high enough that a 0.1 rad tracking error still
# produces ~2 Nm — sufficient to overcome typical BLDC cogging (~0.3 Nm).
_KP_FLOOR = 20.0
_KD_FLOOR = 0.5


class GainPolicy:
    """
    Compute per-joint kp/kd from the mass matrix and a stiffness preset.

    When dynamics are unavailable (pinocchio not installed), falls back
    to fixed scalar gains.

    When constructed with a dynamics_fn, encapsulates adaptive gain
    computation including mass-matrix caching.
    """

    def __init__(self, stiffness: Stiffness = Stiffness.STIFF,
                 fallback_kp: float = 50.0, fallback_kd: float = 1.0,
                 dynamics_fn=None,
                 reflected_inertia=None,
                 cache_interval: int = 10):
        self._stiffness = stiffness
        self._fallback_kp = fallback_kp
        self._fallback_kd = fallback_kd
        self._dynamics_fn = dynamics_fn
        self._reflected_inertia = reflected_inertia
        self._cache_interval = cache_interval
        self._cache_M = None
        self._cache_counter = 0

    @property
    def stiffness(self) -> Stiffness:
        return self._stiffness

    @stiffness.setter
    def stiffness(self, value: Stiffness):
        self._stiffness = value

    def compute(self, mass_matrix: np.ndarray,
                reflected_inertia: np.ndarray | None = None) -> tuple[list[float], list[float]]:
        """
        Compute adaptive kp/kd from current mass matrix.

        Gains are scaled proportionally per-joint so that neither kp nor kd
        exceeds the motor protocol limits (kp_max=500, kd_max=5), preserving
        the damping ratio to avoid oscillation.

        Args:
            mass_matrix: n×n mass matrix M(q).
            reflected_inertia: per-joint reflected motor inertia (I_rotor × ratio²).
                When provided, effective inertia = M_diag + reflected_inertia.

        Returns:
            (kps, kds) per-joint gain lists.
        """
        omega_n = self._stiffness.omega_n
        zeta = self._stiffness.zeta
        M_diag = np.diag(mass_matrix).copy()
        np.maximum(M_diag, 1e-4, out=M_diag)

        if reflected_inertia is not None:
            M_diag += np.asarray(reflected_inertia)

        kps = M_diag * (omega_n ** 2)
        kds = 2.0 * zeta * np.sqrt(kps * M_diag)

        # Scale proportionally to maintain damping ratio within protocol limits
        kp_scale = np.minimum(1.0, _KP_MAX / np.maximum(kps, 1e-10))
        kd_scale = np.minimum(1.0, _KD_MAX / np.maximum(kds, 1e-10))
        scale = np.minimum(kp_scale, kd_scale)
        kps *= scale
        kds *= scale

        # Enforce safety floor. Applied after the ceiling so it wins on the
        # low end; the two regimes are disjoint in practice (floor bumps only
        # tiny-M joints, ceiling scales only huge-M ones).
        below_floor = kps < _KP_FLOOR
        if np.any(below_floor):
            kps[below_floor] = _KP_FLOOR
            # Recompute kd for bumped joints so damping ratio ζ is preserved
            # against the raised kp (kd_crit = 2·ζ·√(kp·M)); otherwise the joint
            # would be under-damped and could ring.
            kds[below_floor] = 2.0 * zeta * np.sqrt(
                _KP_FLOOR * np.maximum(M_diag[below_floor], 1e-6)
            )
        np.maximum(kds, _KD_FLOOR, out=kds)

        return kps.tolist(), kds.tolist()

    def fallback(self, n: int) -> tuple[list[float], list[float]]:
        """Fixed gains when dynamics model is unavailable."""
        return [self._fallback_kp] * n, [self._fallback_kd] * n

    def compute_adaptive(self, q, num_joints: int) -> tuple[list[float], list[float]]:
        """
        Compute adaptive gains for joint configuration q, with mass-matrix caching.

        Requires dynamics_fn to be set at construction. Falls back to fixed gains
        if dynamics are unavailable or computation fails.
        """
        if self._dynamics_fn is None:
            return self.fallback(num_joints)
        try:
            self._cache_counter += 1
            if self._cache_M is None or self._cache_counter >= self._cache_interval:
                dynamics = self._dynamics_fn()
                self._cache_M = dynamics.mass_matrix(q)
                self._cache_counter = 0
            return self.compute(self._cache_M, self._reflected_inertia)
        except ImportError:
            return self.fallback(num_joints)
        except Exception as e:
            logger.warning(f"Adaptive gain computation failed: {e}. Using fixed gains.")
            return self.fallback(num_joints)

    def reset_cache(self) -> None:
        """Reset mass-matrix cache. Call at the start of each new motion."""
        self._cache_M = None
        self._cache_counter = 0
