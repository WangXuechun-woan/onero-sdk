"""
MuJoCo simulation backend — physics-based arm simulation for testing and validation.

Implements ArmBackend using MuJoCo's physics engine. Supports:
- MIT impedance control (PD + torque feedforward)
- Velocity control
- Pure torque control
- Real-time or accelerated simulation

Each control command runs multiple physics substeps with closed-loop PD
recomputation, matching how real motor drivers operate internally at high rate.

Install: pip install onero-sdk[sim]
"""

import os
import time

import numpy as np

try:
    import mujoco
except ImportError:
    mujoco = None

from .backend import ArmBackend, ArmState


class MujocoBackend(ArmBackend):
    """
    MuJoCo-based simulation backend.

    Simulates the arm using full rigid-body dynamics. Each command()
    call runs multiple physics substeps with PD torque recomputed each
    substep (closed-loop), matching real motor driver behavior.
    """

    def __init__(self, urdf_path: str = None, model_name: str = "A1",
                 control_dt: float = 0.005, physics_dt: float = 0.0005,
                 realtime: bool = False, armature: float = 0.01,
                 on_step=None):
        """
        Args:
            urdf_path: Path to URDF file. Auto-detected if None.
            model_name: Model name for auto-detection (default "A1R").
            control_dt: Control loop period (seconds). Default 5ms (200Hz).
            physics_dt: Physics simulation timestep (seconds). Default 0.5ms.
            realtime: If True, sleep to match wall-clock time.
            armature: Reflected rotor inertia per joint (kg⋅m²). Compensates
                      for URDF not modeling motor rotor + gearbox inertia.
            on_step: Optional callback called after each command() step.
                     Use for viewer sync: `on_step=lambda: viewer.sync()`.
        """
        if mujoco is None:
            raise ImportError(
                "MujocoBackend requires mujoco. Install with: pip install onero-sdk[sim]"
            )

        if urdf_path is None:
            urdf_path = self._find_urdf(model_name)

        self._model = mujoco.MjModel.from_xml_path(urdf_path)
        self._model.opt.timestep = physics_dt
        # Add reflected motor inertia (stabilizes simulation for lightweight distal links)
        self._model.dof_armature[:] = armature
        self._data = mujoco.MjData(self._model)
        self._control_dt = control_dt
        self._physics_dt = physics_dt
        self._substeps = max(1, int(round(control_dt / physics_dt)))
        self._realtime = realtime
        self._on_step = on_step
        self._step_count = 0
        self._enabled = False
        self._nq = self._model.nq
        # Torque limits from model registry (peak torque ~2x continuous for real motors)
        # Fall back to a generous simulation limit when URDF has effort=0.
        from .model_registry import get_model
        model_info = get_model(model_name)
        raw_limits = np.array([j.t_max for j in model_info.joints]) * 2.0
        # When URDF doesn't specify effort, use 200 Nm as a safe simulation ceiling
        self._torque_limits = np.where(raw_limits > 0, raw_limits, 200.0)
        # Disable contact checking for faster sim (arm-only, no environment)
        self._model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        # Pre-allocated MjData for gravity computation (avoid per-substep allocation)
        self._grav_data = mujoco.MjData(self._model)
        # Pre-compute adaptive fallback gains from MuJoCo mass matrix at zero config
        # This adapts automatically to updated URDF inertia parameters.
        self._adaptive_gains = {
            "stiff":  self._compute_mujoco_gains(omega_n=40.0, zeta=1.0),
            "medium": self._compute_mujoco_gains(omega_n=20.0, zeta=1.0),
            "soft":   self._compute_mujoco_gains(omega_n=8.0,  zeta=0.7),
        }
        self._adaptive_kps, self._adaptive_kds = self._adaptive_gains["stiff"]

    @staticmethod
    def _find_urdf(model_name: str) -> str:
        from .model_registry import _parse_model_variant
        base_model, variant = _parse_model_variant(model_name)
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(pkg_dir, "urdf", base_model, f"{base_model.lower()}_{variant}.urdf"),
            os.path.join(pkg_dir, "urdf", base_model, f"{base_model.lower()}_r.urdf"),
            os.path.join(pkg_dir, "urdf", base_model, f"{base_model.lower()}_l.urdf"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            f"No URDF found for model '{model_name}'. Searched: {candidates}")

    def _compute_mujoco_gains(self, omega_n: float = 40.0, zeta: float = 1.0,
                               kp_max: float = 500.0, kd_max: float = 5.0,
                               kp_floor: float = 30.0, kd_floor: float = 0.5
                               ) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-joint PD gains from MuJoCo mass matrix at zero config.

        Uses the same impedance formula as GainPolicy but reads M(q) directly
        from MuJoCo, so gains automatically adapt to any URDF inertia update.
        """
        tmp_data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, tmp_data)
        M = np.zeros((self._nq, self._nq))
        mujoco.mj_fullM(self._model, tmp_data, M)
        M_diag = np.maximum(np.diag(M), 1e-4)

        kps = M_diag * omega_n ** 2
        kds = 2.0 * zeta * np.sqrt(kps * M_diag)

        kp_scale = np.minimum(1.0, kp_max / np.maximum(kps, 1e-10))
        kd_scale = np.minimum(1.0, kd_max / np.maximum(kds, 1e-10))
        scale = np.minimum(kp_scale, kd_scale)
        kps *= scale
        kds *= scale

        # Enforce floor so distal lightweight joints still track
        below = kps < kp_floor
        if np.any(below):
            kps[below] = kp_floor
            kds[below] = 2.0 * zeta * np.sqrt(kp_floor * np.maximum(M_diag[below], 1e-6))
        np.maximum(kds, kd_floor, out=kds)
        return kps, kds

    def set_stiffness(self, stiffness: str) -> None:
        """Switch adaptive gain preset (stiff/medium/soft)."""
        if stiffness in self._adaptive_gains:
            self._adaptive_kps, self._adaptive_kds = self._adaptive_gains[stiffness]

    @property
    def num_joints(self) -> int:
        return self._nq

    def enable(self) -> ArmState:
        self._enabled = True
        mujoco.mj_resetData(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)
        # Stabilise at zero config under gravity using stiff PD + gravity comp.
        # Without this, the first state returned reflects free-fall drift and
        # the tracking-error watchdog fires immediately on motion start.
        zero = np.zeros(self._nq)
        kp = self._adaptive_gains["stiff"][0]
        kd = self._adaptive_gains["stiff"][1]
        for _ in range(self._substeps * 20):   # ~100ms of settling
            q = self._data.qpos[:self._nq]
            v = self._data.qvel[:self._nq]
            self._grav_data.qpos[:self._nq] = q
            self._grav_data.qvel[:self._nq] = 0.0
            self._grav_data.qacc[:self._nq] = 0.0
            mujoco.mj_inverse(self._model, self._grav_data)
            tau_grav = self._grav_data.qfrc_inverse[:self._nq]
            tau = kp * (zero - q) + kd * (0.0 - v) + tau_grav
            np.clip(tau, -self._torque_limits, self._torque_limits, out=tau)
            self._data.qfrc_applied[:self._nq] = tau
            mujoco.mj_step(self._model, self._data)
        return self._get_state()

    def disable(self) -> None:
        self._data.qfrc_applied[:self._nq] = 0.0
        self._enabled = False

    def command(self, positions: list[float], velocities: list[float],
                kps: list[float], kds: list[float],
                torques: list[float]) -> ArmState:
        pos_target = np.array(positions)
        vel_target = np.array(velocities)
        # Use MuJoCo-derived adaptive gains when caller provides scalar fallback
        # (kp == default 50.0 means no explicit override from user).
        kp_arr = np.array(kps)
        kd_arr = np.array(kds)
        if np.allclose(kp_arr, 50.0) and np.allclose(kd_arr, 1.0):
            kp_arr = self._adaptive_kps.copy()
            kd_arr = self._adaptive_kds.copy()
        tau_ff = np.array(torques)

        t0 = time.monotonic()
        for _ in range(self._substeps):
            q = self._data.qpos[:self._nq]
            v = self._data.qvel[:self._nq]
            # Compute gravity compensation torque via MuJoCo inverse dynamics
            # with zero acceleration and velocity (pure gravity term).
            self._grav_data.qpos[:self._nq] = q
            self._grav_data.qvel[:self._nq] = 0.0
            self._grav_data.qacc[:self._nq] = 0.0
            mujoco.mj_inverse(self._model, self._grav_data)
            tau_grav = self._grav_data.qfrc_inverse[:self._nq]
            tau = kp_arr * (pos_target - q) + kd_arr * (vel_target - v) + tau_ff + tau_grav
            np.clip(tau, -self._torque_limits, self._torque_limits, out=tau)
            self._data.qfrc_applied[:self._nq] = tau
            mujoco.mj_step(self._model, self._data)

        if self._realtime:
            elapsed = time.monotonic() - t0
            sleep_time = self._control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        self._step_count += 1
        if self._on_step and self._step_count % 10 == 0:
            self._on_step()

        return self._get_state()

    def command_velocity(self, velocities: list[float]) -> ArmState:
        vel_target = np.array(velocities)
        kd_vel = 5.0

        t0 = time.monotonic()
        for _ in range(self._substeps):
            v = self._data.qvel[:self._nq]
            tau = kd_vel * (vel_target - v)
            np.clip(tau, -self._torque_limits, self._torque_limits, out=tau)
            self._data.qfrc_applied[:self._nq] = tau
            mujoco.mj_step(self._model, self._data)

        if self._realtime:
            elapsed = time.monotonic() - t0
            sleep_time = self._control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        self._step_count += 1
        if self._on_step and self._step_count % 10 == 0:
            self._on_step()

        return self._get_state()

    def command_torque(self, torques: list[float]) -> ArmState:
        tau = np.array(torques)
        np.clip(tau, -self._torque_limits, self._torque_limits, out=tau)
        t0 = time.monotonic()
        for _ in range(self._substeps):
            self._data.qfrc_applied[:self._nq] = tau
            mujoco.mj_step(self._model, self._data)

        if self._realtime:
            elapsed = time.monotonic() - t0
            sleep_time = self._control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        self._step_count += 1
        if self._on_step and self._step_count % 10 == 0:
            self._on_step()

        return self._get_state()

    def get_state(self) -> ArmState:
        return self._get_state()

    def close(self) -> None:
        self._enabled = False

    def _get_state(self) -> ArmState:
        return ArmState(
            joints=list(self._data.qpos[:self._nq]),
            velocities=list(self._data.qvel[:self._nq]),
            torques=list(self._data.qfrc_applied[:self._nq]),
            timestamp=self._data.time,
        )

    def reset(self, positions: list[float] = None) -> ArmState:
        """Reset simulation to given joint positions (or zero) and re-stabilise."""
        mujoco.mj_resetData(self._model, self._data)
        if positions is not None:
            self._data.qpos[:self._nq] = np.array(positions)
        mujoco.mj_forward(self._model, self._data)
        # Re-run settling loop so the returned state is at the held position
        target = np.array(positions) if positions is not None else np.zeros(self._nq)
        kp = self._adaptive_gains["stiff"][0]
        kd = self._adaptive_gains["stiff"][1]
        for _ in range(self._substeps * 20):
            q = self._data.qpos[:self._nq]
            v = self._data.qvel[:self._nq]
            self._grav_data.qpos[:self._nq] = q
            self._grav_data.qvel[:self._nq] = 0.0
            self._grav_data.qacc[:self._nq] = 0.0
            mujoco.mj_inverse(self._model, self._grav_data)
            tau_grav = self._grav_data.qfrc_inverse[:self._nq]
            tau = kp * (target - q) + kd * (0.0 - v) + tau_grav
            np.clip(tau, -self._torque_limits, self._torque_limits, out=tau)
            self._data.qfrc_applied[:self._nq] = tau
            mujoco.mj_step(self._model, self._data)
        return self._get_state()
