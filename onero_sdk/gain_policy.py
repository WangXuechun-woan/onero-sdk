"""
Adaptive gain policy — compute per-joint kp/kd from mass matrix and desired dynamics.

Uses the relationship between impedance parameters and system dynamics:
  kp_i = M_ii(q) * ωn²
  kd_i = 2 * ζ * sqrt(kp_i * M_ii(q))

This ensures consistent dynamic response across all configurations:
the arm feels equally "stiff" or "soft" regardless of pose.
"""

from enum import Enum

import numpy as np


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


class GainPolicy:
    """
    Compute per-joint kp/kd from the mass matrix and a stiffness preset.

    When dynamics are unavailable (pinocchio not installed), falls back
    to fixed scalar gains.
    """

    def __init__(self, stiffness: Stiffness = Stiffness.STIFF,
                 fallback_kp: float = 50.0, fallback_kd: float = 1.0):
        self._stiffness = stiffness
        self._fallback_kp = fallback_kp
        self._fallback_kd = fallback_kd

    @property
    def stiffness(self) -> Stiffness:
        return self._stiffness

    @stiffness.setter
    def stiffness(self, value: Stiffness):
        self._stiffness = value

    def compute(self, mass_matrix: np.ndarray) -> tuple[list[float], list[float]]:
        """
        Compute adaptive kp/kd from current mass matrix.

        Args:
            mass_matrix: n×n mass matrix M(q).

        Returns:
            (kps, kds) per-joint gain lists.
        """
        omega_n = self._stiffness.omega_n
        zeta = self._stiffness.zeta
        M_diag = np.diag(mass_matrix).copy()
        np.maximum(M_diag, 1e-4, out=M_diag)
        kps = M_diag * (omega_n ** 2)
        kds = 2.0 * zeta * np.sqrt(kps * M_diag)
        return kps.tolist(), kds.tolist()

    def fallback(self, n: int) -> tuple[list[float], list[float]]:
        """Fixed gains when dynamics model is unavailable."""
        return [self._fallback_kp] * n, [self._fallback_kd] * n
