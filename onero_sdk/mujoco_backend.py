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
        from .model_registry import get_model
        model_info = get_model(model_name)
        self._torque_limits = np.array([j.t_max for j in model_info.joints]) * 2.0
        # Disable contact checking for faster sim (arm-only, no environment)
        self._model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

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

    @property
    def num_joints(self) -> int:
        return self._nq

    def enable(self) -> ArmState:
        self._enabled = True
        mujoco.mj_resetData(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)
        return self._get_state()

    def disable(self) -> None:
        self._data.qfrc_applied[:self._nq] = 0.0
        self._enabled = False

    def command(self, positions: list[float], velocities: list[float],
                kps: list[float], kds: list[float],
                torques: list[float]) -> ArmState:
        pos_target = np.array(positions)
        vel_target = np.array(velocities)
        kp = np.array(kps)
        kd = np.array(kds)
        tau_ff = np.array(torques)

        t0 = time.monotonic()
        for _ in range(self._substeps):
            q = self._data.qpos[:self._nq]
            v = self._data.qvel[:self._nq]
            tau = kp * (pos_target - q) + kd * (vel_target - v) + tau_ff
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

    def close(self, disable: bool = True) -> None:
        if disable:
            self._data.qfrc_applied[:self._nq] = 0.0
        self._enabled = False

    def _get_state(self) -> ArmState:
        return ArmState(
            joints=list(self._data.qpos[:self._nq]),
            velocities=list(self._data.qvel[:self._nq]),
            torques=list(self._data.qfrc_applied[:self._nq]),
            timestamp=self._data.time,
        )

    def reset(self, positions: list[float] = None) -> ArmState:
        """Reset simulation to given joint positions (or zero)."""
        mujoco.mj_resetData(self._model, self._data)
        if positions is not None:
            self._data.qpos[:self._nq] = np.array(positions)
        mujoco.mj_forward(self._model, self._data)
        return self._get_state()
