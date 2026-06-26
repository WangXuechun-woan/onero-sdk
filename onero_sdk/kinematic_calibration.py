"""
Kinematic calibration — correct URDF parameters for improved absolute accuracy.

Uses the "single-point constraint" method: the user touches a fixed reference
point from multiple configurations. Since all configurations should produce the
same end-effector position, we optimize small corrections to the kinematic
parameters (joint origin offsets) to minimize the position spread.

No external equipment required — just a fixed reference point (e.g. table corner).

Install: pip install onero-sdk (uses scipy for optimization)
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

try:
    from scipy.optimize import least_squares
except ImportError:
    least_squares = None

from .fk import URDFChain, _axis_angle_matrix


@dataclass
class KinematicCalibrationResult:
    """Result of kinematic calibration."""
    corrections: np.ndarray
    residual_mm: float
    num_touches: int
    centroid: np.ndarray
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def save(self, path: str) -> None:
        data = {
            "format": "onero_kinematic_cal_v1",
            "corrections": self.corrections.tolist(),
            "residual_mm": self.residual_mm,
            "num_touches": self.num_touches,
            "centroid": self.centroid.tolist(),
            "timestamp": self.timestamp,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "KinematicCalibrationResult":
        with open(path, 'r') as f:
            data = json.load(f)
        if data.get("format") != "onero_kinematic_cal_v1":
            raise ValueError(f"Unknown format: {data.get('format')}")
        return cls(
            corrections=np.array(data["corrections"]),
            residual_mm=data["residual_mm"],
            num_touches=data["num_touches"],
            centroid=np.array(data["centroid"]),
            timestamp=data["timestamp"],
        )


class KinematicCalibrator:
    """
    Single-point kinematic calibration.

    Usage:
        calibrator = KinematicCalibrator(arm)

        # Touch the same fixed point from different arm configurations
        for i in range(20):
            input(f"Touch reference point (config {i+1}/20), press Enter")
            calibrator.record_touch()

        # Solve for corrections
        result = calibrator.solve()
        print(f"Residual after calibration: {result.residual_mm:.2f} mm")

        # Apply corrections (affects all subsequent FK/IK)
        arm.apply_kinematic_calibration(result)

        # Save for later use
        result.save("kinematic_cal.json")
    """

    def __init__(self, fk_chain: URDFChain):
        """
        Args:
            fk_chain: The URDFChain to calibrate.
        """
        if least_squares is None:
            raise ImportError(
                "KinematicCalibrator requires scipy. "
                "Install with: pip install scipy"
            )
        self._fk = fk_chain
        self._touches: list[np.ndarray] = []

    @classmethod
    def from_arm(cls, arm) -> "KinematicCalibrator":
        """Create calibrator from an Arm instance."""
        fk = arm._get_fk()
        return cls(fk)

    @property
    def num_touches(self) -> int:
        return len(self._touches)

    def record_touch(self, q: Optional[list[float]] = None, arm=None) -> np.ndarray:
        """
        Record a touch configuration.

        Args:
            q: Joint angles. If None, reads from arm.
            arm: Arm instance to read current joints from (if q is None).

        Returns:
            End-effector position computed from this configuration.
        """
        if q is None:
            if arm is None:
                raise ValueError("Must provide either q or arm")
            q = list(arm.state.joints)
        q = list(q)
        self._touches.append(np.array(q))
        pos, _ = self._fk.get_ee_pose(q)
        return pos

    def clear(self) -> None:
        """Clear all recorded touches."""
        self._touches.clear()

    def get_spread_mm(self) -> float:
        """Get current position spread (before calibration) in mm."""
        if len(self._touches) < 2:
            return 0.0
        positions = np.array([self._fk.get_ee_pose(q)[0] for q in self._touches])
        centroid = positions.mean(axis=0)
        errors = positions - centroid
        return float(np.sqrt(np.mean(np.sum(errors ** 2, axis=1)))) * 1000.0

    def solve(self, max_correction_mm: float = 5.0) -> KinematicCalibrationResult:
        """
        Solve for kinematic parameter corrections.

        Optimizes small xyz offsets for each joint origin to minimize the
        spread of end-effector positions across all recorded touches.

        Args:
            max_correction_mm: Maximum allowed correction per parameter (mm).
                              Prevents unrealistic solutions.

        Returns:
            KinematicCalibrationResult with corrections and residual.
        """
        if len(self._touches) < 4:
            raise ValueError(
                f"Need at least 4 touch configurations, got {len(self._touches)}. "
                "More configurations (15-30) give better results."
            )

        n_joints = self._fk.num_joints
        # Parameters: xyz offset for each joint origin (3 * n_joints)
        n_params = 3 * n_joints

        bounds_m = max_correction_mm / 1000.0

        def residuals(delta):
            positions = []
            for q in self._touches:
                pos = self._fk_with_corrections(q, delta.reshape(n_joints, 3))
                positions.append(pos)
            positions = np.array(positions)
            centroid = positions.mean(axis=0)
            errors = (positions - centroid).ravel()
            return errors

        result = least_squares(
            residuals,
            x0=np.zeros(n_params),
            bounds=(-bounds_m, bounds_m),
            method='trf',
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )

        corrections = result.x.reshape(n_joints, 3)

        # Compute final residual
        positions = []
        for q in self._touches:
            pos = self._fk_with_corrections(q, corrections)
            positions.append(pos)
        positions = np.array(positions)
        centroid = positions.mean(axis=0)
        errors = positions - centroid
        residual_mm = float(np.sqrt(np.mean(np.sum(errors ** 2, axis=1)))) * 1000.0

        return KinematicCalibrationResult(
            corrections=corrections,
            residual_mm=residual_mm,
            num_touches=len(self._touches),
            centroid=centroid,
        )

    def _fk_with_corrections(self, q: list[float], corrections: np.ndarray) -> np.ndarray:
        """Compute FK position with joint origin corrections applied."""
        T = np.eye(4)
        for i, (origin_T, axis) in enumerate(self._fk._joints):
            T_corrected = origin_T.copy()
            T_corrected[:3, 3] = T_corrected[:3, 3] + corrections[i]
            T = T @ T_corrected
            R_joint = np.eye(4)
            R_joint[:3, :3] = _axis_angle_matrix(axis, q[i])
            T = T @ R_joint
        T = T @ self._fk._ee_fixed_transform
        return T[:3, 3]


def apply_calibration(fk_chain: URDFChain, result: KinematicCalibrationResult) -> None:
    """
    Apply kinematic calibration corrections to an FK chain (in-place).

    After this call, all subsequent FK computations use the corrected parameters.

    Args:
        fk_chain: The URDFChain to modify.
        result: Calibration result from KinematicCalibrator.solve().
    """
    corrections = result.corrections
    if corrections.shape[0] != fk_chain.num_joints:
        raise ValueError(
            f"Correction has {corrections.shape[0]} joints, "
            f"but chain has {fk_chain.num_joints}"
        )
    for i, (origin_T, axis) in enumerate(fk_chain._joints):
        origin_T[:3, 3] += corrections[i]
