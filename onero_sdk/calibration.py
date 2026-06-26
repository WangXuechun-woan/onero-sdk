"""
Dynamics parameter identification — calibrate inertial and friction parameters.

Uses pinocchio's joint torque regressor (τ = Y·π) to build observation matrices,
then solves with truncated SVD for numerically stable identification of the
identifiable (base) parameter subspace.

Supports:
- Static (gravity-only) identification: mass, center-of-mass
- Dynamic identification with friction: adds Coulomb + viscous per joint

Install: pip install onero-sdk[dynamics]
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

try:
    import pinocchio as pin
except ImportError:
    pin = None


@dataclass
class CalibrationResult:
    """Identified dynamics parameters with metadata."""
    params: np.ndarray
    residual_rms: float
    num_configs: int
    rank: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    friction_coulomb: Optional[np.ndarray] = None
    friction_viscous: Optional[np.ndarray] = None

    def save(self, path: str) -> None:
        data = {
            "format": "onero_calibration_v1",
            "params": self.params.tolist(),
            "residual_rms": self.residual_rms,
            "num_configs": self.num_configs,
            "rank": self.rank,
            "timestamp": self.timestamp,
        }
        if self.friction_coulomb is not None:
            data["friction_coulomb"] = self.friction_coulomb.tolist()
        if self.friction_viscous is not None:
            data["friction_viscous"] = self.friction_viscous.tolist()
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CalibrationResult":
        with open(path, 'r') as f:
            data = json.load(f)
        if data.get("format") != "onero_calibration_v1":
            raise ValueError(f"Unknown calibration format: {data.get('format')}")
        fc = np.array(data["friction_coulomb"]) if "friction_coulomb" in data else None
        fv = np.array(data["friction_viscous"]) if "friction_viscous" in data else None
        return cls(
            params=np.array(data["params"]),
            residual_rms=data["residual_rms"],
            num_configs=data["num_configs"],
            rank=data["rank"],
            timestamp=data["timestamp"],
            friction_coulomb=fc,
            friction_viscous=fv,
        )


class GravityCalibrator:
    """
    Dynamics parameter identification using pinocchio's torque regressor.

    Given N configurations with measured torques, identifies the dynamics
    parameters (mass, center-of-mass, inertia) and optionally friction
    coefficients that best explain the data. Uses truncated SVD to handle
    rank-deficient parameter spaces (only base parameters are identifiable).
    """

    def __init__(self, model, data):
        if pin is None:
            raise ImportError(
                "GravityCalibrator requires pinocchio. "
                "Install with: pip install onero-sdk[dynamics]"
            )
        self._model = model
        self._data = data
        self._nq = model.nq

    def generate_configs(self, num_configs: int,
                         joint_limits: list[tuple[float, float]],
                         margin: float = 0.1,
                         seed: Optional[int] = None) -> list[list[float]]:
        """
        Generate calibration configurations using Latin Hypercube Sampling.

        Provides better coverage of the joint space than pure random sampling
        with the same number of points.

        Args:
            num_configs: Number of configurations to generate.
            joint_limits: (min, max) for each joint.
            margin: Fraction of range to stay away from limits.
            seed: Random seed for reproducibility.

        Returns:
            List of joint configurations.
        """
        rng = np.random.default_rng(seed)
        n_joints = len(joint_limits)
        configs = []

        # Latin Hypercube: divide each dimension into num_configs equal intervals
        for j in range(n_joints):
            lo, hi = joint_limits[j]
            span = hi - lo
            lo_safe = lo + margin * span
            hi_safe = hi - margin * span
            intervals = np.linspace(lo_safe, hi_safe, num_configs + 1)
            # One random sample per interval
            col = np.array([
                rng.uniform(intervals[i], intervals[i + 1])
                for i in range(num_configs)
            ])
            rng.shuffle(col)
            configs.append(col)

        return np.column_stack(configs).tolist()

    def fit(self, configs: list[list[float]],
            torques: list[list[float]],
            svd_threshold: float = 1e-6) -> CalibrationResult:
        """
        Fit dynamics parameters from static configuration/torque pairs.

        Args:
            configs: N joint configurations, each shape (nq,).
            torques: N measured static torques, each shape (nq,).
            svd_threshold: Singular value threshold relative to max.
                          Values below this are treated as zero (unidentifiable).

        Returns:
            CalibrationResult with identified parameters.
        """
        configs_arr = np.asarray(configs, dtype=float)
        torques_arr = np.asarray(torques, dtype=float)
        n_samples = len(configs_arr)

        # Build stacked regressor
        Y_rows = []
        for q in configs_arr:
            Y = pin.computeJointTorqueRegressor(
                self._model, self._data, q,
                np.zeros(self._nq), np.zeros(self._nq)
            )
            Y_rows.append(Y)
        Y_stacked = np.vstack(Y_rows)
        tau_stacked = torques_arr.ravel()

        # Truncated SVD solve
        U, S, Vt = np.linalg.svd(Y_stacked, full_matrices=False)
        tol = S[0] * svd_threshold
        rank = int(np.sum(S > tol))

        S_inv = np.zeros_like(S)
        S_inv[:rank] = 1.0 / S[:rank]
        params = Vt.T @ (S_inv * (U.T @ tau_stacked))

        # Compute residual
        tau_pred = Y_stacked @ params
        residual = tau_stacked - tau_pred
        residual_rms = float(np.sqrt(np.mean(residual ** 2)))

        return CalibrationResult(
            params=params,
            residual_rms=residual_rms,
            num_configs=n_samples,
            rank=rank,
        )

    def fit_with_friction(self, configs: list[list[float]],
                          torques: list[list[float]],
                          velocities: list[list[float]],
                          svd_threshold: float = 1e-6) -> CalibrationResult:
        """
        Fit dynamics parameters plus per-joint friction model.

        Friction model: τ_friction = fc * sign(v) + fv * v
        Requires data collected at non-zero velocities (slow motion).

        Args:
            configs: N joint configurations.
            torques: N measured torques (during motion).
            velocities: N joint velocities at measurement time.
            svd_threshold: SVD truncation threshold.

        Returns:
            CalibrationResult with params + friction coefficients.
        """
        configs_arr = np.asarray(configs, dtype=float)
        torques_arr = np.asarray(torques, dtype=float)
        vels_arr = np.asarray(velocities, dtype=float)
        n_samples = len(configs_arr)
        nq = self._nq

        # Build augmented regressor: [Y_dynamics | sign(v) | v]
        Y_rows = []
        for i in range(n_samples):
            Y_dyn = pin.computeJointTorqueRegressor(
                self._model, self._data, configs_arr[i],
                vels_arr[i], np.zeros(nq)
            )
            # Friction columns: per-joint Coulomb and viscous
            sign_v = np.diag(np.sign(vels_arr[i]))  # (nq, nq)
            visc_v = np.diag(vels_arr[i])            # (nq, nq)
            Y_aug = np.hstack([Y_dyn, sign_v, visc_v])
            Y_rows.append(Y_aug)

        Y_stacked = np.vstack(Y_rows)
        tau_stacked = torques_arr.ravel()

        # Truncated SVD solve
        U, S, Vt = np.linalg.svd(Y_stacked, full_matrices=False)
        tol = S[0] * svd_threshold
        rank = int(np.sum(S > tol))

        S_inv = np.zeros_like(S)
        S_inv[:rank] = 1.0 / S[:rank]
        params_full = Vt.T @ (S_inv * (U.T @ tau_stacked))

        # Split: dynamics params | coulomb | viscous
        n_dyn = Y_rows[0].shape[1] - 2 * nq
        dyn_params = params_full[:n_dyn]
        fc = params_full[n_dyn:n_dyn + nq]
        fv = params_full[n_dyn + nq:]

        # Residual
        tau_pred = Y_stacked @ params_full
        residual = tau_stacked - tau_pred
        residual_rms = float(np.sqrt(np.mean(residual ** 2)))

        return CalibrationResult(
            params=dyn_params,
            residual_rms=residual_rms,
            num_configs=n_samples,
            rank=rank,
            friction_coulomb=fc,
            friction_viscous=fv,
        )
