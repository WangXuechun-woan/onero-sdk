"""
Dual-arm MuJoCo simulation — two arms mounted on a human-like torso.

Uses MjSpec.attach() to compose a single physics scene from two URDF arms
and a procedural torso body. Provides unified PD control for both arms
simultaneously within one MuJoCo simulation.

Install: pip install onero-sdk[sim]
"""

import os
import time
import threading

import numpy as np

try:
    import mujoco
except ImportError:
    mujoco = None

from .backend import ArmBackend, ArmState


class _DualMujocoArmBackend(ArmBackend):
    """One ArmBackend view into a shared DualMujocoBackend scene."""

    def __init__(self, parent, side: str):
        self._parent = parent
        self._side = side
        self._enabled = False

    @property
    def num_joints(self) -> int:
        return 7

    def enable(self) -> ArmState:
        self._enabled = True
        return self.get_state()

    def disable(self) -> None:
        self._enabled = False

    def command(self, positions, velocities, kps, kds, torques) -> ArmState:
        return self._parent._command_side(
            self._side,
            positions=list(positions), velocities=list(velocities),
            kps=list(kps), kds=list(kds), torques=list(torques),
        )

    def command_velocity(self, velocities) -> ArmState:
        state = self.get_state()
        return self.command(
            state.joints, velocities, [0.0] * 7, [5.0] * 7, [0.0] * 7)

    def command_torque(self, torques) -> ArmState:
        state = self.get_state()
        return self.command(
            state.joints, [0.0] * 7, [0.0] * 7, [0.0] * 7, torques)

    def get_state(self) -> ArmState:
        return self._parent._get_side_state(self._side)

    def close(self, disable: bool = False) -> None:
        if disable:
            self.disable()


class DualMujocoBackend:
    """
    Dual-arm MuJoCo simulation with a human-like torso.

    Builds a composite scene: ground plane + torso (box + head) + left arm + right arm.
    Both arms are controlled via PD impedance in a single physics simulation.

    Joint ordering: right arm [0:7], left arm [7:14].
    """

    def __init__(self, model_name: str = "A1R", realtime: bool = True,
                 control_dt: float = 0.005, physics_dt: float = 0.0005,
                 armature: float = 0.01):
        if mujoco is None:
            raise ImportError(
                "DualMujocoBackend requires mujoco. Install with: pip install mujoco"
            )

        self._nq = 7
        self._control_dt = control_dt
        self._physics_dt = physics_dt
        self._substeps = max(1, int(round(control_dt / physics_dt)))
        self._realtime = realtime
        self._viewer = None

        self._model, self._data = self._build_scene(model_name, physics_dt, armature)

        # Pair commands from the two public Arm instances into one atomic
        # MuJoCo control cycle. This lets DualArm exercise the same Arm motion,
        # gain, gravity and safety code used by hardware without racing mjData.
        self._command_condition = threading.Condition()
        self._pending_commands = {}
        self._command_cycle = 0
        self._cycle_results = {}
        self._cycle_error = None
        self._control_cycles = 0
        self.on_step = None
        self._left_arm_backend = _DualMujocoArmBackend(self, "left")
        self._right_arm_backend = _DualMujocoArmBackend(self, "right")

        from .model_registry import get_model
        model_info = get_model(model_name)
        self._torque_limits = np.array([j.t_max for j in model_info.joints]) * 2.0

    def _build_scene(self, model_name: str, physics_dt: float, armature: float):
        """Build composite MuJoCo scene with torso + two arms."""
        from .model_registry import _parse_model_variant
        base_model, _ = _parse_model_variant(model_name)
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_dir = os.path.join(pkg_dir, "urdf", base_model)
        r_urdf = os.path.join(urdf_dir, f"{base_model.lower()}_r.urdf")
        l_urdf = os.path.join(urdf_dir, f"{base_model.lower()}_l.urdf")

        spec = mujoco.MjSpec()
        spec.compiler.meshdir = urdf_dir

        # Ground plane
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[2.0, 2.0, 0.01],
            rgba=[0.9, 0.9, 0.9, 1.0],
        )

        # Lighting
        spec.worldbody.add_light(pos=[0.5, 0.5, 2.5], dir=[-0.3, -0.3, -1])
        spec.worldbody.add_light(pos=[-0.5, -0.5, 2.5], dir=[0.3, 0.3, -1])

        # Table in front of robot (simulates desk-edge mounting)
        table = spec.worldbody.add_body(name='table', pos=[0.55, 0, 0.72])
        # Table top
        table.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.30, 0.40, 0.02],
            rgba=[0.6, 0.45, 0.3, 1.0],
        )
        # Table legs
        for ly, lx in [(-0.25, -0.25), (-0.25, 0.25), (0.25, -0.25), (0.25, 0.25)]:
            table.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[0.02, 0.35],
                pos=[lx, ly, -0.37],
                rgba=[0.5, 0.35, 0.2, 1.0],
            )

        # Torso body (fixed in world — seated adult proportions)
        torso = spec.worldbody.add_body(name='torso', pos=[0, 0, 0.82])
        # Main body
        torso.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.10, 0.06, 0.28],
            rgba=[0.35, 0.45, 0.55, 1.0],
        )
        # Front chest marker (red plate on +X face = front)
        torso.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.005, 0.04, 0.06],
            pos=[0.105, 0, 0.05],
            rgba=[0.85, 0.2, 0.2, 1.0],
        )
        # Right shoulder marker (blue dot)
        torso.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.025],
            pos=[0, -0.09, 0.22],
            rgba=[0.2, 0.4, 0.9, 1.0],
        )
        # Left shoulder marker (green dot)
        torso.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.025],
            pos=[0, 0.09, 0.22],
            rgba=[0.2, 0.8, 0.3, 1.0],
        )
        # Head
        torso.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.07],
            pos=[0, 0, 0.36],
            rgba=[0.75, 0.65, 0.55, 1.0],
        )
        # Neck
        torso.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.03, 0.04],
            pos=[0, 0, 0.30],
            rgba=[0.75, 0.65, 0.55, 1.0],
        )

        # Use the same authoritative R_world_base matrices as Arm/DualArm;
        # do not maintain an independent set of visually tuned Euler angles.
        from .mounting import (
            R_WORLD_BASE_HUMANOID_LEFT,
            R_WORLD_BASE_HUMANOID_RIGHT,
        )

        def _mujoco_quat(rotation):
            quat = np.empty(4, dtype=float)  # MuJoCo order: [w, x, y, z]
            mujoco.mju_mat2Quat(quat, np.asarray(rotation, dtype=float).ravel())
            return quat

        r_site = torso.add_site(
            name='r_shoulder', pos=[0, -0.12, 0.22],
            quat=_mujoco_quat(R_WORLD_BASE_HUMANOID_RIGHT),
        )
        l_site = torso.add_site(
            name='l_shoulder', pos=[0, 0.12, 0.22],
            quat=_mujoco_quat(R_WORLD_BASE_HUMANOID_LEFT),
        )

        # Load and attach arms
        r_arm_spec = mujoco.MjSpec.from_file(r_urdf)
        l_arm_spec = mujoco.MjSpec.from_file(l_urdf)

        spec.attach(r_arm_spec, prefix='R_', site=r_site)
        spec.attach(l_arm_spec, prefix='L_', site=l_site)

        # Compile
        model = spec.compile()
        model.opt.timestep = physics_dt
        model.dof_armature[:] = armature
        # Collision filtering via contype/conaffinity bitmasks:
        # - Arm geoms: contype=2, conaffinity=1 → arms don't collide with each other
        # - Table geoms: contype=1, conaffinity=2 → table collides with arms
        # - Torso geoms: contype=0, conaffinity=0 → no collision
        # Check: arm(2) vs arm(aff=1): 2&1=0, arm(2) vs arm(aff=1): 0 → no collision
        # Check: arm(2) vs table(aff=2): 2&2=2 → collision ✓
        torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
        table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'table')
        for i in range(model.ngeom):
            bid = model.geom_bodyid[i]
            if bid == torso_id:
                model.geom_contype[i] = 0
                model.geom_conaffinity[i] = 0
            elif bid == table_id:
                model.geom_contype[i] = 1
                model.geom_conaffinity[i] = 2
            else:
                # Arm links and ground
                model.geom_contype[i] = 2
                model.geom_conaffinity[i] = 1
        data = mujoco.MjData(model)

        return model, data

    @property
    def control_cycles(self) -> int:
        """Number of paired public-Arm control cycles executed."""
        return self._control_cycles

    def arm_backends(self) -> tuple[ArmBackend, ArmBackend]:
        """Return ``(left, right)`` ArmBackend views of the shared scene."""
        return self._left_arm_backend, self._right_arm_backend

    def _get_side_state(self, side: str) -> ArmState:
        with self._command_condition:
            return self._get_left_state() if side == "left" else self._get_right_state()

    def _command_side(self, side: str, **command) -> ArmState:
        """Barrier-pair one command from each Arm into an atomic physics step."""
        with self._command_condition:
            cycle = self._command_cycle
            if side in self._pending_commands:
                raise RuntimeError(f"duplicate {side} command in dual control cycle")
            self._pending_commands[side] = command

            if len(self._pending_commands) == 2:
                try:
                    right, left = self._execute_impedance_pair(
                        self._pending_commands["right"],
                        self._pending_commands["left"],
                    )
                    self._cycle_results = {"right": right, "left": left}
                    self._cycle_error = None
                except BaseException as exc:
                    self._cycle_results = {}
                    self._cycle_error = exc
                finally:
                    self._pending_commands.clear()
                    self._command_cycle += 1
                    self._command_condition.notify_all()
            else:
                completed = self._command_condition.wait_for(
                    lambda: self._command_cycle != cycle, timeout=10.0)
                if not completed:
                    self._pending_commands.pop(side, None)
                    raise TimeoutError(
                        "timed out waiting for the other arm's control frame")

            if self._cycle_error is not None:
                raise self._cycle_error
            return self._cycle_results[side]

    def _execute_impedance_pair(self, right, left):
        targets = np.asarray(right["positions"] + left["positions"], dtype=float)
        velocities = np.asarray(right["velocities"] + left["velocities"], dtype=float)
        kps = np.asarray(right["kps"] + left["kps"], dtype=float)
        kds = np.asarray(right["kds"] + left["kds"], dtype=float)
        torques = np.asarray(right["torques"] + left["torques"], dtype=float)
        torque_lim = np.concatenate([self._torque_limits, self._torque_limits])

        t0 = time.monotonic()
        for _ in range(self._substeps):
            q = self._data.qpos[:14]
            v = self._data.qvel[:14]
            tau = kps * (targets - q) + kds * (velocities - v) + torques
            np.clip(tau, -torque_lim, torque_lim, out=tau)
            self._data.qfrc_applied[:14] = tau
            mujoco.mj_step(self._model, self._data)

        if self._realtime:
            sleep_time = self._control_dt - (time.monotonic() - t0)
            if sleep_time > 0:
                time.sleep(sleep_time)
        if self._viewer is not None:
            self._viewer.sync()

        self._control_cycles += 1
        if self.on_step is not None:
            self.on_step(self)
        return self._get_right_state(), self._get_left_state()

    def reset(self, right_pos=None, left_pos=None):
        """Reset simulation to given joint positions."""
        mujoco.mj_resetData(self._model, self._data)
        if right_pos is not None:
            self._data.qpos[:7] = np.asarray(right_pos)
        if left_pos is not None:
            self._data.qpos[7:14] = np.asarray(left_pos)
        mujoco.mj_forward(self._model, self._data)

    def step(self, right_target, left_target,
             kp: float = 150.0, kd: float = 5.0,
             right_tau_ff=None, left_tau_ff=None) -> tuple[ArmState, ArmState]:
        """
        Execute one control step for both arms simultaneously.

        Args:
            right_target: Target joint positions for right arm (7,).
            left_target: Target joint positions for left arm (7,).
            kp: Position gain.
            kd: Velocity damping gain.
            right_tau_ff: Feedforward torque for right arm (optional).
            left_tau_ff: Feedforward torque for left arm (optional).

        Returns:
            Tuple of (right_state, left_state).
        """
        r_target = np.asarray(right_target, dtype=float)
        l_target = np.asarray(left_target, dtype=float)

        targets = np.concatenate([r_target, l_target])
        tau_ff = np.zeros(14)
        if right_tau_ff is not None:
            tau_ff[:7] = np.asarray(right_tau_ff)
        if left_tau_ff is not None:
            tau_ff[7:14] = np.asarray(left_tau_ff)

        torque_lim = np.concatenate([self._torque_limits, self._torque_limits])

        t0 = time.monotonic()
        for _ in range(self._substeps):
            q = self._data.qpos[:14]
            v = self._data.qvel[:14]
            tau = kp * (targets - q) + kd * (0.0 - v) + tau_ff
            np.clip(tau, -torque_lim, torque_lim, out=tau)
            self._data.qfrc_applied[:14] = tau
            mujoco.mj_step(self._model, self._data)

        if self._realtime:
            elapsed = time.monotonic() - t0
            sleep_time = self._control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        if self._viewer is not None:
            self._viewer.sync()

        return self._get_right_state(), self._get_left_state()

    def movej(self, right_target, left_target, duration: float = 2.0,
              kp: float = 150.0, kd: float = 5.0):
        """
        Move both arms to target positions with linear interpolation.

        Args:
            right_target: Target joint positions for right arm (7,).
            left_target: Target joint positions for left arm (7,).
            duration: Motion duration in seconds.
            kp: Position gain.
            kd: Velocity damping gain.
        """
        r_target = np.asarray(right_target, dtype=float)
        l_target = np.asarray(left_target, dtype=float)
        r_start = self._data.qpos[:7].copy()
        l_start = self._data.qpos[7:14].copy()

        n_steps = max(1, int(duration / self._control_dt))
        for i in range(n_steps):
            alpha = (i + 1) / n_steps
            r_cmd = r_start + alpha * (r_target - r_start)
            l_cmd = l_start + alpha * (l_target - l_start)
            self.step(r_cmd, l_cmd, kp=kp, kd=kd)

    def hold(self, duration: float = 1.0, kp: float = 150.0, kd: float = 5.0):
        """Hold current position for a duration."""
        r_pos = self._data.qpos[:7].copy()
        l_pos = self._data.qpos[7:14].copy()
        n_steps = max(1, int(duration / self._control_dt))
        for _ in range(n_steps):
            self.step(r_pos, l_pos, kp=kp, kd=kd)

    def _get_right_state(self) -> ArmState:
        return ArmState(
            joints=list(self._data.qpos[:7]),
            velocities=list(self._data.qvel[:7]),
            torques=list(self._data.qfrc_applied[:7]),
            timestamp=self._data.time,
        )

    def _get_left_state(self) -> ArmState:
        return ArmState(
            joints=list(self._data.qpos[7:14]),
            velocities=list(self._data.qvel[7:14]),
            torques=list(self._data.qfrc_applied[7:14]),
            timestamp=self._data.time,
        )
