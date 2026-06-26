"""
Arm — the core primitive of onero SDK.

Provides high-level motion API (movej/movel/movep/servoj/servop) on top of
a pluggable backend. Users interact with this class; the backend handles
whether commands go to real hardware or simulation.
"""

import math
import logging
import threading
from typing import Callable, Optional, Union

import numpy as np

from .backend import ArmBackend, ArmState
from .model_registry import get_model, ArmModel
from .safety import SafetyConfig, SafetyMonitor
from .exceptions import OneroError, OneroConfigError
from .end_effector import EndEffector
from .trajectory import Recorder, Recording, TrajectoryFrame, TrajectoryPlanner, Waypoint
from . import geometry
from .clock import ControlClock
from .motion_executor import MotionExecutor, MotionFrame, VelocityFrame
from .safe_backend import SafeBackend
from .gain_policy import GainPolicy, Stiffness

logger = logging.getLogger(__name__)


class Arm:
    """
    Robotic arm controller (N-DOF, determined by model registry).

    Create with a model name + CAN channel (real hardware):
        arm = Arm("A1", channel="can0")

    Or with a custom backend (simulation, etc.):
        arm = Arm("A1", backend=my_backend)

    Or from a config file:
        arm = Arm.from_config("arm.yaml")
    """

    def __init__(self, model: str = "A1", channel: Optional[str] = None,
                 backend: Optional[ArmBackend] = None,
                 safety: Optional[SafetyConfig] = None,
                 end_effector: Optional[EndEffector] = None,
                 on_error: Optional[Callable] = None):
        """
        Args:
            model: Arm model name (e.g. "A1"). Looked up in model registry.
            channel: CAN channel name (e.g. "can0"). Creates a CAN backend.
                     Mutually exclusive with `backend`.
            backend: Custom backend instance. Mutually exclusive with `channel`.
            safety: Safety configuration. Default limits active if None.
            end_effector: Optional gripper or dexterous hand attached to this arm.
            on_error: Callback invoked on critical errors. Signature: (OneroError) -> None.
        """
        self._model_info = get_model(model)
        self._on_error = on_error
        self._enabled = False

        # Create backend
        if backend is not None and channel is not None:
            raise OneroConfigError("Cannot specify both 'channel' and 'backend'")

        if backend is not None:
            self._backend = backend
        elif channel is not None:
            from .can_backend import CanBackend
            self._backend = CanBackend(
                channel, self._model_info.joints,
                on_miss=self._on_can_miss,
                on_fault=self._on_can_fault,
            )
        else:
            raise OneroConfigError("Must specify either 'channel' or 'backend'")

        # Safety monitor
        joint_limits = []
        for i, spec in enumerate(self._model_info.joints):
            p_min = spec.p_min if spec.p_min is not None else -spec.p_max
            p_max = spec.p_max
            if p_min > p_max:
                raise OneroConfigError(
                    f"Joint {i}: invalid limits p_min={p_min} > p_max={p_max}")
            joint_limits.append((p_min, p_max))
        self._safety = SafetyMonitor(
            num_joints=self._model_info.num_joints,
            joint_limits=joint_limits,
            config=safety or SafetyConfig(),
            on_error=on_error,
            on_stop=lambda: self._backend.disable(),
        )

        # Wrap backend with safety layer (transparent clamping)
        self._safe_backend = SafeBackend(self._backend, self._safety)

        # Cached state
        self._state = ArmState(
            joints=[0.0] * self._model_info.num_joints,
            velocities=[0.0] * self._model_info.num_joints,
            torques=[0.0] * self._model_info.num_joints,
        )

        # FK (lazy init)
        self._fk = None
        self._ik = None
        self._dynamics = None

        # End effector (optional)
        self._end_effector = end_effector

        # Recording
        self._recorder = Recorder()

        # Freedrive state
        self._freedrive_active = False
        self._freedrive_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()  # protects _state and backend access
        self._clock = ControlClock()

        # Motion executor (unified control loop — uses safe backend)
        self._executor = MotionExecutor(
            backend=self._safe_backend,
            safety=self._safety,
            recorder=self._recorder,
            clock=self._clock,
            get_state=lambda: self._state,
            set_state=lambda s: setattr(self, '_state', s),
            gravity_torques_fn=self._gravity_torques,
        )

        # Force mode state
        self._force_mode_origin: Optional[list[float]] = None

        # Control parameters from model
        self._default_kp = self._model_info.default_kp
        self._default_kd = self._model_info.default_kd
        self._control_freq = self._model_info.control_freq
        self._gain_policy = GainPolicy(
            stiffness=Stiffness.STIFF,
            fallback_kp=self._default_kp,
            fallback_kd=self._default_kd,
        )

    # ─── Properties ───────────────────────────────────────────────────────────

    @property
    def model(self) -> ArmModel:
        return self._model_info

    @property
    def num_joints(self) -> int:
        return self._model_info.num_joints

    @property
    def control_freq(self) -> float:
        """Control loop frequency (Hz). Use for timing your servo loops."""
        return self._control_freq

    @property
    def state(self) -> ArmState:
        """Last known arm state (cached from most recent command/query)."""
        return self._state

    @property
    def joints(self) -> list[float]:
        """Current joint positions (rad). Shortcut for state.joints."""
        return self._state.joints

    @property
    def is_moving(self) -> bool:
        """True if any joint velocity exceeds 0.01 rad/s."""
        return any(abs(v) > 0.01 for v in self._state.velocities)

    @property
    def is_freedrive(self) -> bool:
        """True if freedrive mode is active."""
        return self._freedrive_active

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def end_effector(self) -> Optional[EndEffector]:
        """Attached end effector (gripper or hand), or None."""
        return self._end_effector

    @property
    def gripper(self) -> Optional[EndEffector]:
        """Shortcut for end_effector (convenience alias)."""
        return self._end_effector

    @property
    def on_error(self) -> Optional[Callable]:
        return self._on_error

    @on_error.setter
    def on_error(self, handler: Optional[Callable]):
        self._on_error = handler
        self._safety.on_error = handler

    # ─── Lifecycle ────────────────────────────────────────────────────────────

    def enable(self) -> ArmState:
        """
        Enable all joints. Auto-clears errors and retries once.
        Returns initial arm state.

        Raises:
            OneroHardwareError: if any joint fails to enable after retry.
        """
        self._state = self._backend.enable()
        self._enabled = True
        self._safety.reset(self._state.joints)
        if self._end_effector is not None:
            self._end_effector.enable()
        logger.info(f"Arm enabled. Initial joints: {self._state.joints}")
        return self._state

    def disable(self) -> None:
        """Disable all joints."""
        if self._end_effector is not None:
            self._end_effector.disable()
        self._backend.disable()
        self._enabled = False
        logger.info("Arm disabled")

    def close(self) -> None:
        """Disable arm and release all resources (CAN socket, end effector)."""
        try:
            # Stop freedrive thread before disabling to avoid socket I/O on closed bus
            if self._freedrive_active:
                self.freedrive_disable()
            if self._enabled:
                self.disable()
        finally:
            if self._end_effector is not None:
                try:
                    self._end_effector.shutdown()
                except Exception:
                    pass
            self._backend.close()

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, *args):
        self.close()

    # ─── State Query ──────────────────────────────────────────────────────────

    def get_state(self) -> ArmState:
        """Actively query current state from hardware."""
        self._state = self._backend.get_state()
        return self._state

    def get_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute current end-effector pose via FK.

        Returns:
            (position [3], rotation_matrix [3x3])

        Raises:
            OneroConfigError: if URDF not configured for this model.
        """
        fk = self._get_fk()
        return fk.get_ee_pose(self._state.joints)

    # ─── Planned Motion (blocking) ───────────────────────────────────────────

    def movej(self, joints: list[float], duration: Optional[float] = None,
              speed: Optional[float] = None, acceleration: Optional[float] = None,
              profile: str = 'auto',
              kp: Optional[float] = None, kd: Optional[float] = None) -> ArmState:
        """
        Move to target joint angles with smooth interpolation.

        Blocks until motion is complete.

        Args:
            joints: Target angles for each joint (rad).
            duration: Motion duration (seconds). Auto-computed if None.
            speed: Max joint speed (rad/s). Alternative to duration —
                   duration is computed from the largest joint displacement.
                   Ignored if duration is explicitly set.
            acceleration: Joint acceleration (rad/s²). Used with speed to
                         compute duration via trapezoidal profile estimation.
                         Default 1.4 rad/s² (UR-compatible).
            profile: Interpolation profile type:
                     'auto' — trapezoidal (recommended, industry standard)
                     'cosine' — smooth cosine S-curve (legacy behavior)
                     'trapezoidal' — constant acceleration (UR/KUKA style)
                     's_curve' — jerk-limited 7-segment profile (smoothest)
            kp: Position gain (default from model).
            kd: Damping gain (default from model).

        Returns:
            Final arm state.
        """
        self._check_enabled()
        self._check_no_freedrive()
        self._begin_motion()
        use_adaptive = (kp is None and kd is None)
        if not use_adaptive:
            kp = self._default_kp if kp is None else kp
            kd = self._default_kd if kd is None else kd
        n = self.num_joints

        if len(joints) != n:
            raise OneroConfigError(f"Expected {n} joints, got {len(joints)}")

        start_pos = list(self._state.joints)
        accel = acceleration if acceleration is not None else 1.4

        # Resolve profile
        if profile == 'auto':
            profile = 'trapezoidal'

        if duration is None:
            max_delta = max(abs(joints[i] - start_pos[i]) for i in range(n))
            if speed is not None:
                duration = self._duration_from_speed(max_delta, speed, accel)
            else:
                duration = max(1.0, max_delta / 1.0)

        steps = max(1, int(duration * self._control_freq))
        dt = 1.0 / self._control_freq

        if profile == 'trapezoidal' or profile == 's_curve':
            from .trajectory_profiles import (
                TrapezoidalProfile, SCurveProfile, multi_joint_trapezoidal,
                compute_synchronized_duration,
            )
            v_max_per = speed if speed is not None else 2.0
            v_maxs = [v_max_per] * n
            a_maxs = [accel] * n

            # Compute synchronized profiles for all joints
            deltas = [joints[i] - start_pos[i] for i in range(n)]
            sync_dur = compute_synchronized_duration(deltas, v_maxs, a_maxs)
            actual_dur = max(duration, sync_dur)

            if profile == 's_curve':
                j_max = accel * 10.0
                profiles = []
                for i in range(n):
                    profiles.append(SCurveProfile(deltas[i], v_max_per, accel, j_max))
                # Use the longest S-curve duration
                s_dur = max(p.duration for p in profiles)
                actual_dur = max(actual_dur, s_dur)
                actual_steps = max(1, int(actual_dur / dt))

                def _frames():
                    for step in range(actual_steps + 1):
                        t = step * dt
                        positions = []
                        velocities = []
                        for i in range(n):
                            pt = profiles[i].evaluate(min(t, profiles[i].duration))
                            positions.append(start_pos[i] + pt.position)
                            velocities.append(pt.velocity)
                        if use_adaptive:
                            kps, kds = self._compute_gains(positions)
                        else:
                            kps, kds = [kp] * n, [kd] * n
                        yield MotionFrame(
                            positions=positions, velocities=velocities,
                            kps=kps, kds=kds, torques=[0.0] * n, dt=dt,
                        )
            else:
                # Trapezoidal
                joint_profiles = multi_joint_trapezoidal(
                    start_pos, joints, v_maxs, a_maxs, dt,
                    target_duration=actual_dur)
                num_frames = len(joint_profiles[0]) if joint_profiles else steps + 1

                def _frames():
                    for step in range(num_frames):
                        positions = [joint_profiles[i][step].position for i in range(n)]
                        velocities = [joint_profiles[i][step].velocity for i in range(n)]
                        if use_adaptive:
                            kps, kds = self._compute_gains(positions)
                        else:
                            kps, kds = [kp] * n, [kd] * n
                        yield MotionFrame(
                            positions=positions, velocities=velocities,
                            kps=kps, kds=kds, torques=[0.0] * n, dt=dt,
                        )
        else:
            # Cosine interpolation (default, original behavior)
            def _frames():
                for step in range(steps + 1):
                    t = step / steps
                    alpha = 0.5 * (1.0 - math.cos(math.pi * t))
                    vel_alpha = 0.5 * math.pi * math.sin(math.pi * t) / duration
                    positions = [start_pos[i] + alpha * (joints[i] - start_pos[i]) for i in range(n)]
                    velocities = [vel_alpha * (joints[i] - start_pos[i]) for i in range(n)]
                    if use_adaptive:
                        kps, kds = self._compute_gains(positions)
                    else:
                        kps, kds = [kp] * n, [kd] * n
                    yield MotionFrame(
                        positions=positions, velocities=velocities,
                        kps=kps, kds=kds, torques=[0.0] * n, dt=dt,
                    )

        self._state = self._executor.run(_frames())

        if not self._safety.emergency_stopped:
            max_err = max(abs(self._state.joints[i] - joints[i]) for i in range(n))
            if max_err > 0.01:
                logger.warning(
                    f"movej: target clamped by safety limits (max error: {max_err:.3f} rad)")

        return self._state

    def movel(self, position: Union[list, np.ndarray],
              rotation: Union[list, np.ndarray] = None,
              duration: Optional[float] = None,
              speed: Optional[float] = None, acceleration: Optional[float] = None,
              kp: Optional[float] = None, kd: Optional[float] = None) -> ArmState:
        """
        Move end-effector along a straight line in Cartesian space.

        Interpolates linearly in position and uses SLERP for rotation.
        Solves IK at each step.

        Args:
            position: Target EE position [x, y, z] (meters), or a 6D pose
                      [x, y, z, rx, ry, rz] (meters + rotation vector) when
                      rotation is not provided separately.
            rotation: Target EE rotation (3x3 matrix, quaternion [x,y,z,w],
                      or rotvec [rx,ry,rz]). Optional if position is 6D.
            duration: Motion duration (seconds). Auto-computed if None.
            speed: Max tool speed (m/s). Alternative to duration —
                   duration is computed from the Cartesian distance.
                   Ignored if duration is explicitly set.
            acceleration: Tool acceleration (m/s²). Used with speed to
                         compute duration. Default 1.2 m/s² (UR-compatible).
            kp: Position gain.
            kd: Damping gain.

        Returns:
            Final arm state.
        """
        self._check_enabled()
        self._check_no_freedrive()
        self._begin_motion()
        ik = self._get_ik()
        use_adaptive = (kp is None and kd is None)
        if not use_adaptive:
            kp = self._default_kp if kp is None else kp
            kd = self._default_kd if kd is None else kd
        n = self.num_joints

        # Parse pose: accept [x,y,z,rx,ry,rz] or separate position+rotation
        target_pos, target_rot = self._parse_pose("movel", position, rotation)

        # Get current EE pose
        fk = self._get_fk()
        start_pos_ee, start_rot_ee = fk.get_ee_pose(self._state.joints)

        # Auto-compute duration
        if duration is None:
            dist = np.linalg.norm(target_pos - start_pos_ee)
            if speed is not None:
                accel = acceleration if acceleration is not None else 1.2
                duration = self._duration_from_speed(dist, speed, accel)
            else:
                duration = max(1.0, dist / 0.1)  # ~0.1 m/s max

        steps = max(1, int(duration * self._control_freq))
        dt = 1.0 / self._control_freq
        pos_delta = target_pos - start_pos_ee

        def _frames():
            prev_joints = None
            for step in range(steps + 1):
                t = step / steps
                alpha = 0.5 * (1.0 - math.cos(math.pi * t))
                interp_pos = start_pos_ee + alpha * pos_delta
                interp_rot = geometry.slerp_matrix(start_rot_ee, target_rot, alpha)

                q = ik.solve(interp_pos, interp_rot, dt=dt)
                joints = q.tolist() if hasattr(q, 'tolist') else list(q)

                if prev_joints is not None:
                    velocities = [(joints[i] - prev_joints[i]) / dt for i in range(n)]
                else:
                    velocities = [0.0] * n
                prev_joints = list(joints)

                if use_adaptive:
                    kps, kds = self._compute_gains(joints)
                else:
                    kps, kds = [kp] * n, [kd] * n
                yield MotionFrame(
                    positions=joints, velocities=velocities,
                    kps=kps, kds=kds, torques=[0.0] * n, dt=dt,
                )

        self._state = self._executor.run(_frames())

        return self._state

    def movep(self, position: Union[list, np.ndarray],
              rotation: Union[list, np.ndarray] = None,
              duration: Optional[float] = None,
              speed: Optional[float] = None, acceleration: Optional[float] = None,
              kp: Optional[float] = None, kd: Optional[float] = None) -> ArmState:
        """
        Move to target Cartesian pose (path not constrained to straight line).

        Solves IK for target pose, then performs joint-space interpolation.
        Faster than movel but EE path may be curved.

        Args:
            position: Target EE position [x, y, z] (meters), or a 6D pose
                      [x, y, z, rx, ry, rz] when rotation is not provided.
            rotation: Target EE rotation (3x3 matrix, quaternion [x,y,z,w],
                      or rotvec [rx,ry,rz]). Optional if position is 6D.
            duration: Motion duration (seconds).
            speed: Max joint speed (rad/s). Alternative to duration.
            acceleration: Joint acceleration (rad/s²). Default 1.4.
            kp: Position gain.
            kd: Damping gain.

        Returns:
            Final arm state.
        """
        self._check_enabled()
        self._check_no_freedrive()
        ik = self._get_ik()

        target_pos, target_rot = self._parse_pose("movep", position, rotation)

        # Solve IK for target
        q_target = ik.solve(target_pos, target_rot, dt=1.0 / self._control_freq)
        target_joints = q_target.tolist() if hasattr(q_target, 'tolist') else list(q_target)

        # Joint-space interpolation to that target
        return self.movej(target_joints, duration=duration, speed=speed,
                          acceleration=acceleration, kp=kp, kd=kd)

    def movec(self, via_pose: Union[list, np.ndarray],
              target_pose: Union[list, np.ndarray],
              duration: Optional[float] = None,
              speed: Optional[float] = None, acceleration: Optional[float] = None,
              kp: Optional[float] = None, kd: Optional[float] = None) -> ArmState:
        """
        Move end-effector along a circular arc defined by three points.

        The arc passes through: current pose → via point → target pose.
        Useful for welding, gluing, polishing along curved surfaces.

        Args:
            via_pose: Intermediate point [x, y, z, rx, ry, rz] (meters, rad).
                      Rotation as axis-angle (rotation vector).
            target_pose: Final point [x, y, z, rx, ry, rz] (meters, rad).
            duration: Motion duration (seconds). Auto-computed if None.
            speed: Max tool speed (m/s). Alternative to duration.
            acceleration: Tool acceleration (m/s²). Default 1.2.
            kp: Position gain.
            kd: Damping gain.

        Returns:
            Final arm state.
        """
        self._check_enabled()
        self._check_no_freedrive()
        self._begin_motion()
        ik = self._get_ik()
        fk = self._get_fk()
        use_adaptive = (kp is None and kd is None)
        if not use_adaptive:
            kp = self._default_kp if kp is None else kp
            kd = self._default_kd if kd is None else kd
        n = self.num_joints

        via_pose = np.asarray(via_pose, dtype=float)
        target_pose = np.asarray(target_pose, dtype=float)

        # Get three points: start (current), via, target
        p0, r0 = fk.get_ee_pose(self._state.joints)
        p1 = via_pose[:3]
        p2 = target_pose[:3]
        r1 = geometry.rotvec_to_matrix(via_pose[3:])
        r2 = geometry.rotvec_to_matrix(target_pose[3:])

        # Fit circle through 3 points
        center, radius, normal = geometry.fit_circle_3points(p0, p1, p2)

        # Auto-compute duration from arc length
        if duration is None:
            arc_length = geometry.arc_length_3points(p0, p1, p2, center, radius, normal)
            if speed is not None:
                accel = acceleration if acceleration is not None else 1.2
                duration = self._duration_from_speed(arc_length, speed, accel)
            else:
                duration = max(1.0, arc_length / 0.1)  # ~0.1 m/s

        steps = max(1, int(duration * self._control_freq))
        dt = 1.0 / self._control_freq

        # Parameterize arc: angle from p0 to p2 through p1
        v0 = p0 - center
        total_angle = geometry.signed_arc_angle(v0, p2 - center, normal, p1 - center)

        def _frames():
            for step in range(steps + 1):
                t = step / steps
                alpha = 0.5 * (1.0 - math.cos(math.pi * t))
                angle = alpha * total_angle

                interp_pos = center + geometry.rotate_vector(v0, normal, angle)

                if alpha <= 0.5:
                    t_seg = alpha * 2
                    Ra = geometry.slerp_matrix(r0, r1, t_seg)
                    Rb = geometry.slerp_matrix(r1, r2, t_seg * 0.5)
                else:
                    t_seg = (alpha - 0.5) * 2
                    Ra = geometry.slerp_matrix(r0, r1, 0.5 + t_seg * 0.5)
                    Rb = geometry.slerp_matrix(r1, r2, t_seg)
                interp_rot = geometry.slerp_matrix(Ra, Rb, alpha)

                q = ik.solve(interp_pos, interp_rot, dt=dt)
                joints = q.tolist() if hasattr(q, 'tolist') else list(q)

                if use_adaptive:
                    kps, kds = self._compute_gains(joints)
                else:
                    kps, kds = [kp] * n, [kd] * n
                yield MotionFrame(
                    positions=joints, velocities=[0.0] * n,
                    kps=kps, kds=kds, torques=[0.0] * n, dt=dt,
                )

        self._state = self._executor.run(_frames())
        return self._state

    def home(self, duration: Optional[float] = None,
             kp: Optional[float] = None, kd: Optional[float] = None) -> ArmState:
        """
        Move all joints to zero position.

        Uses kp=30.0 by default (gentler than normal motion) to reduce
        overshoot when homing from large angles. Pass kp explicitly to override.
        """
        kp = kp if kp is not None else 30.0
        return self.movej([0.0] * self.num_joints, duration=duration or 3.0, kp=kp, kd=kd)

    # ─── Real-time Control (non-blocking, single frame) ──────────────────────

    def servoj(self, joints: list[float],
               kp: Optional[float] = None, kd: Optional[float] = None,
               tau_ff: Optional[list[float]] = None) -> ArmState:
        """
        Send a single joint-space target (real-time control).

        Call this at control frequency (e.g. 100Hz). Safety limits are
        applied by default (rate clamping, joint limits).

        Args:
            joints: Target angles for each joint (rad).
            kp: Position gain (default from model). Set to 0 for pure torque mode.
            kd: Damping gain (default from model). Set to 0 for pure torque mode.
            tau_ff: Feedforward torques (Nm). Added to PD output.
                    If None, gravity compensation is used.

        Returns:
            Current arm state from feedback.
        """
        self._check_enabled()
        self._check_no_freedrive()
        kp = self._default_kp if kp is None else kp
        kd = self._default_kd if kd is None else kd
        n = self.num_joints

        if len(joints) != n:
            raise OneroConfigError(f"Expected {n} joints, got {len(joints)}")

        if tau_ff is not None:
            torques = list(tau_ff)
        else:
            torques = self._gravity_torques(list(self._state.joints))
        self._state = self._safe_backend.command(
            positions=joints,
            velocities=[0.0] * n,
            kps=[kp] * n,
            kds=[kd] * n,
            torques=torques,
        )
        self._recorder.record_frame(
            self._state.joints, self._state.velocities, self._state.torques)
        return self._state

    def servop(self, position: Union[list, np.ndarray],
               rotation: Union[list, np.ndarray] = None,
               kp: Optional[float] = None, kd: Optional[float] = None) -> ArmState:
        """
        Send a single task-space target (real-time control).

        Solves IK incrementally then calls servoj. Call at control frequency.

        Args:
            position: Target EE position [x, y, z] (meters), or a 6D pose
                      [x, y, z, rx, ry, rz] when rotation is not provided.
            rotation: Target EE rotation (3x3 matrix, quaternion [x,y,z,w],
                      or rotvec [rx,ry,rz]). Optional if position is 6D.
            kp: Position gain.
            kd: Damping gain.

        Returns:
            Current arm state from feedback.
        """
        self._check_enabled()
        ik = self._get_ik()

        target_pos, target_rot = self._parse_pose("servop", position, rotation)

        q = ik.solve(target_pos, target_rot, dt=1.0 / self._control_freq)
        joints = q.tolist() if hasattr(q, 'tolist') else list(q)

        return self.servoj(joints, kp=kp, kd=kd)

    # ─── Velocity Control ────────────────────────────────────────────────────

    def speedj(self, velocities: list[float], acceleration: float = 1.0,
               duration: Optional[float] = None) -> ArmState:
        """
        Move at target joint velocities with trapezoidal acceleration profile.

        Ramps up from current velocity to target, holds, then ramps down to zero.
        Blocking call — returns after motion completes.

        Args:
            velocities: Target velocity per joint (rad/s).
            acceleration: Acceleration/deceleration magnitude (rad/s^2).
            duration: Total motion time (seconds). If None, holds velocity
                      for 1 second after ramp-up then decelerates to stop.

        Returns:
            Final arm state.
        """
        self._check_enabled()
        self._check_no_freedrive()
        self._begin_motion()
        n = self.num_joints
        dt = 1.0 / self._control_freq

        if len(velocities) != n:
            raise OneroConfigError(f"Expected {n} velocities, got {len(velocities)}")

        target_vel = list(velocities)

        max_vel_delta = max(abs(v) for v in target_vel) if target_vel else 0
        ramp_time = max_vel_delta / acceleration if acceleration > 0 else 0
        ramp_steps = int(ramp_time * self._control_freq)

        if duration is not None:
            hold_time = max(0, duration - 2 * ramp_time)
        else:
            hold_time = 1.0
        hold_steps = int(hold_time * self._control_freq)

        def _frames():
            # Phase 1: Ramp up
            for step in range(ramp_steps):
                alpha = (step + 1) / max(ramp_steps, 1)
                yield VelocityFrame(velocities=[alpha * v for v in target_vel], dt=dt)
            # Phase 2: Hold
            for step in range(hold_steps):
                yield VelocityFrame(velocities=target_vel, dt=dt)
            # Phase 3: Ramp down
            for step in range(ramp_steps):
                alpha = 1.0 - (step + 1) / max(ramp_steps, 1)
                yield VelocityFrame(velocities=[alpha * v for v in target_vel], dt=dt)

        self._state = self._executor.run(_frames())

        # Final: zero velocity, then switch back to position hold
        self._state = self._command_velocity_safe([0.0] * n)
        g = self._gravity_torques(list(self._state.joints))
        self._state = self._backend.command(
            positions=list(self._state.joints),
            velocities=[0.0] * n,
            kps=[self._default_kp] * n,
            kds=[self._default_kd] * n,
            torques=g,
        )
        self._safety.reconcile_position(list(self._state.joints))
        return self._state

    def speedl(self, velocities: list[float], acceleration: float = 0.5,
               duration: Optional[float] = None) -> ArmState:
        """
        Move end-effector at target Cartesian velocity.

        Recomputes Jacobian each frame to maintain accurate Cartesian velocity
        as the arm configuration changes.

        Args:
            velocities: Target EE velocity [vx, vy, vz, wx, wy, wz] (m/s, rad/s).
            acceleration: Joint acceleration limit (rad/s^2).
            duration: Total motion time (seconds). Default 1s hold after ramp-up.

        Returns:
            Final arm state.
        """
        self._check_enabled()
        self._check_no_freedrive()
        self._begin_motion()
        dynamics = self._get_dynamics()
        twist = np.asarray(velocities, dtype=float)
        if twist.shape != (6,):
            raise OneroConfigError(
                f"speedl: expected 6-element twist [vx,vy,vz,wx,wy,wz], got shape {twist.shape}")
        n = self.num_joints
        dt = 1.0 / self._control_freq

        # Compute initial joint velocity for ramp timing
        joint_vel_target = dynamics.jacobian_solve(self._state.joints, twist)
        max_vel_delta = float(np.max(np.abs(joint_vel_target)))
        ramp_time = max_vel_delta / acceleration if acceleration > 0 else 0
        ramp_steps = int(ramp_time * self._control_freq)

        if duration is not None:
            hold_time = max(0, duration - 2 * ramp_time)
        else:
            hold_time = 1.0
        hold_steps = int(hold_time * self._control_freq)

        def _frames():
            # Phase 1: Ramp up
            for step in range(ramp_steps):
                alpha = (step + 1) / max(ramp_steps, 1)
                joint_vel = dynamics.jacobian_solve(self._state.joints, alpha * twist)
                yield VelocityFrame(velocities=joint_vel.tolist(), dt=dt)
            # Phase 2: Hold at target velocity
            for step in range(hold_steps):
                joint_vel = dynamics.jacobian_solve(self._state.joints, twist)
                yield VelocityFrame(velocities=joint_vel.tolist(), dt=dt)
            # Phase 3: Ramp down
            for step in range(ramp_steps):
                alpha = 1.0 - (step + 1) / max(ramp_steps, 1)
                joint_vel = dynamics.jacobian_solve(self._state.joints, alpha * twist)
                yield VelocityFrame(velocities=joint_vel.tolist(), dt=dt)

        self._state = self._executor.run(_frames())

        self._state = self._command_velocity_safe([0.0] * n)
        g = self._gravity_torques(list(self._state.joints))
        self._state = self._backend.command(
            positions=list(self._state.joints),
            velocities=[0.0] * n,
            kps=[self._default_kp] * n,
            kds=[self._default_kd] * n,
            torques=g,
        )
        self._safety.reconcile_position(list(self._state.joints))
        return self._state

    def stop(self, deceleration: float = 2.0) -> ArmState:
        """
        Decelerate to zero velocity and hold position.

        Args:
            deceleration: Rate of deceleration (rad/s^2).

        Returns:
            Final arm state.
        """
        self._check_enabled()
        self._check_no_freedrive()
        self._begin_motion()
        n = self.num_joints
        dt = 1.0 / self._control_freq

        # Get fresh state for accurate deceleration
        self._state = self._backend.get_state()
        current_vel = list(self._state.velocities)
        max_vel = max(abs(v) for v in current_vel) if current_vel else 0

        if max_vel < 1e-6:
            return self._state

        ramp_steps = int((max_vel / deceleration) * self._control_freq)

        def _frames():
            for step in range(ramp_steps):
                alpha = 1.0 - (step + 1) / max(ramp_steps, 1)
                yield VelocityFrame(velocities=[alpha * v for v in current_vel], dt=dt)

        self._state = self._executor.run(_frames())

        self._state = self._command_velocity_safe([0.0] * n)
        g = self._gravity_torques(list(self._state.joints))
        self._state = self._backend.command(
            positions=list(self._state.joints),
            velocities=[0.0] * n,
            kps=[self._default_kp] * n,
            kds=[self._default_kd] * n,
            torques=g,
        )
        self._safety.reconcile_position(list(self._state.joints))
        return self._state

    # ─── Recording / Replay ──────────────────────────────────────────────────

    def start_recording(self) -> None:
        """Start recording arm state at every control frame (all motion commands)."""
        self._recorder.start()

    def stop_recording(self) -> Recording:
        """Stop recording and return the captured trajectory."""
        return self._recorder.stop()

    def replay(self, recording: Recording, speed_factor: float = 1.0,
               kp: Optional[float] = None, kd: Optional[float] = None,
               lead_in: Optional[float] = None) -> ArmState:
        """
        Replay a recorded trajectory (blocking).

        If the current position differs from the recording's start, a smooth
        movej transition is performed first to avoid sudden jumps.

        Args:
            recording: Previously recorded trajectory.
            speed_factor: Playback speed multiplier (>1.0 = faster).
            kp: Position gain (default from model).
            kd: Damping gain (default from model).
            lead_in: Duration (seconds) for the transition to recording start.
                     None = auto-calculate from distance (1s per rad max error).
                     0 = skip transition (immediate jump).

        Returns:
            Final arm state.
        """
        self._check_enabled()
        self._check_no_freedrive()

        if recording.num_frames < 2:
            return self._state

        start_joints = recording.frames[0].joints
        max_diff = max(abs(self._state.joints[i] - start_joints[i])
                       for i in range(self.num_joints))

        if lead_in != 0 and max_diff > 0.01:
            if lead_in is None:
                lead_in = max(0.5, min(3.0, max_diff * 1.0))
            self.movej(start_joints, duration=lead_in, kp=kp, kd=kd)

        self._begin_motion()

        use_adaptive = (kp is None and kd is None)
        if not use_adaptive:
            kp = self._default_kp if kp is None else kp
            kd = self._default_kd if kd is None else kd
        n = self.num_joints
        control_dt = 1.0 / self._control_freq

        def _gen():
            frames = recording.frames
            t0 = frames[0].timestamp
            total_dur = frames[-1].timestamp - t0
            playback_dur = total_dur / speed_factor
            num_steps = max(1, int(playback_dur / control_dt))

            for step in range(1, num_steps + 1):
                rec_t = t0 + step * control_dt * speed_factor
                lo = 0
                hi = len(frames) - 1
                while lo < hi - 1:
                    mid = (lo + hi) // 2
                    if frames[mid].timestamp <= rec_t:
                        lo = mid
                    else:
                        hi = mid
                f0, f1 = frames[lo], frames[hi]
                seg_dt = f1.timestamp - f0.timestamp
                alpha = (rec_t - f0.timestamp) / seg_dt if seg_dt > 0 else 0.0
                alpha = min(1.0, max(0.0, alpha))
                joints = [
                    f0.joints[j] + alpha * (f1.joints[j] - f0.joints[j])
                    for j in range(n)
                ]
                vels = [
                    (f0.velocities[j] + alpha * (f1.velocities[j] - f0.velocities[j]))
                    * speed_factor
                    if f0.velocities and f1.velocities else 0.0
                    for j in range(n)
                ]
                if use_adaptive:
                    kps, kds = self._compute_gains(joints)
                else:
                    kps, kds = [kp] * n, [kd] * n
                yield MotionFrame(
                    positions=joints, velocities=vels,
                    kps=kps, kds=kds, torques=[0.0] * n, dt=control_dt,
                )

        self._state = self._executor.run(_gen())
        return self._state

    # ─── Trajectory Execution (blending) ─────────────────────────────────────

    def execute_trajectory(self, frames: list[TrajectoryFrame],
                           kp: Optional[float] = None, kd: Optional[float] = None) -> ArmState:
        """
        Execute a pre-computed trajectory frame by frame (blocking).

        Use TrajectoryPlanner to compute frames with blending, then pass here.

        Args:
            frames: List of trajectory frames from TrajectoryPlanner.compute().
            kp: Position gain.
            kd: Damping gain.

        Returns:
            Final arm state.
        """
        self._check_enabled()
        self._begin_motion()
        use_adaptive = (kp is None and kd is None)
        if not use_adaptive:
            kp = self._default_kp if kp is None else kp
            kd = self._default_kd if kd is None else kd
        n = self.num_joints

        def _gen():
            for frame in frames:
                if use_adaptive:
                    kps, kds = self._compute_gains(frame.joints)
                else:
                    kps, kds = [kp] * n, [kd] * n
                yield MotionFrame(
                    positions=frame.joints,
                    velocities=frame.velocities if hasattr(frame, 'velocities') else [0.0] * n,
                    kps=kps, kds=kds, torques=[0.0] * n, dt=frame.dt,
                )

        self._state = self._executor.run(_gen())
        return self._state

    # ─── Freedrive (Teach Mode) ──────────────────────────────────────────────

    def freedrive_enable(self, damping: float = 0.5,
                         free_axes: Optional[list[int]] = None) -> None:
        """
        Enter freedrive (teach) mode.

        Computes gravity compensation torques and sends as feedforward,
        with kp=0 so the arm can be manually guided. Runs in a background
        thread at control frequency.

        Requires pinocchio: pip install onero-sdk[dynamics]

        Args:
            damping: Joint damping gain (kd) during freedrive.
                     Range: 0.0 (no resistance, arm feels weightless) to
                     ~5.0 (very stiff, hard to move). Default 0.5 is light
                     resistance suitable for demonstration recording.
            free_axes: Which Cartesian axes are free to move [x,y,z,rx,ry,rz].
                      1=free, 0=locked. Default None means all axes free
                      (equivalent to [1,1,1,1,1,1]). Example: [0,0,1,0,0,0]
                      allows only Z-axis movement (useful for surface following).
        """
        self._check_enabled()
        if self._freedrive_active:
            return

        dynamics = self._get_dynamics()
        self._freedrive_active = True

        # Precompute lock mask for constrained-axis freedrive
        lock_mask = None
        if free_axes is not None:
            lock_mask = np.array([0 if a else 1 for a in free_axes], dtype=float)
            if lock_mask.shape != (6,):
                raise OneroConfigError(
                    f"free_axes must have 6 elements [x,y,z,rx,ry,rz], got {len(free_axes)}")

        def _freedrive_loop():
            dt = 1.0 / self._control_freq
            n = self.num_joints
            clock = ControlClock()
            q_lock = None
            kp_lock = 200.0
            while self._freedrive_active:
                if self._safety.emergency_stopped:
                    break
                with self._lock:
                    q = list(self._state.joints)

                gravity_torques = dynamics.gravity_torques(q)
                torques = list(gravity_torques)

                # If some axes are locked, apply stiff constraint via Jacobian
                if lock_mask is not None:
                    if q_lock is None:
                        q_lock = np.array(q)
                    J = dynamics.jacobian(q)
                    x_err = J @ (np.array(q) - q_lock)
                    F_lock = -kp_lock * lock_mask * x_err
                    tau_lock = J.T @ F_lock
                    torques = (np.array(torques) + tau_lock).tolist()

                with self._lock:
                    self._state = self._backend.command(
                        positions=q,
                        velocities=[0.0] * n,
                        kps=[0.0] * n,
                        kds=[damping] * n,
                        torques=torques,
                    )
                    self._recorder.record_frame(
                        self._state.joints, self._state.velocities, self._state.torques)
                clock.tick(dt)

        self._freedrive_thread = threading.Thread(target=_freedrive_loop, daemon=True)
        self._freedrive_thread.start()
        logger.info("Freedrive mode enabled")

    def freedrive_disable(self) -> None:
        """Exit freedrive mode. Arm holds current position."""
        if not self._freedrive_active:
            return
        self._freedrive_active = False
        if self._freedrive_thread is not None:
            self._freedrive_thread.join(timeout=1.0)
            if self._freedrive_thread.is_alive():
                logger.warning("Freedrive thread did not exit within timeout")
            self._freedrive_thread = None
        # Lock position so arm doesn't droop under gravity
        n = self.num_joints
        g = self._gravity_torques(list(self._state.joints))
        self._state = self._backend.command(
            positions=list(self._state.joints),
            velocities=[0.0] * n,
            kps=[self._default_kp] * n,
            kds=[self._default_kd] * n,
            torques=g,
        )
        # Reconcile safety baseline with actual post-freedrive position
        self._safety.reconcile_position(list(self._state.joints))
        logger.info("Freedrive mode disabled")

    def freedrive_step(self, damping: float = 0.5) -> ArmState:
        """
        Single-frame freedrive (for users who manage their own loop).

        Args:
            damping: Joint damping (kd).

        Returns:
            Current arm state.
        """
        self._check_enabled()
        dynamics = self._get_dynamics()
        n = self.num_joints
        q = list(self._state.joints)
        gravity_torques = dynamics.gravity_torques(q)

        self._state = self._backend.command(
            positions=q,
            velocities=[0.0] * n,
            kps=[0.0] * n,
            kds=[damping] * n,
            torques=gravity_torques,
        )
        self._recorder.record_frame(
            self._state.joints, self._state.velocities, self._state.torques)
        return self._state

    # ─── Force Mode ──────────────────────────────────────────────────────────

    def force_mode(self, selection: list[int], wrench: list[float],
                   task_frame: Optional[list[float]] = None,
                   limits: Optional[list[float]] = None,
                   kp: Optional[float] = None, kd: Optional[float] = None) -> ArmState:
        """
        Single-frame hybrid force/position control.

        Applies force in selected Cartesian directions while maintaining
        position in others. Call at control frequency in your own loop.

        Args:
            selection: 6-element binary vector [fx, fy, fz, tx, ty, tz].
                       1 = force control in that direction, 0 = position control.
            wrench: Desired wrench [fx, fy, fz, tx, ty, tz] (N, Nm).
            task_frame: Reference frame for force application [x, y, z, rx, ry, rz].
                        None = base frame. Only the rotation component (rx, ry, rz)
                        is used to transform the wrench and selection directions;
                        the position component is ignored (wrench is applied at EE).
            limits: Speed limits [vx, vy, vz, wx, wy, wz] (m/s, rad/s) for
                    force-controlled directions. Safety cap to prevent runaway.
                    None = no additional speed limiting.
            kp: Position gain for position-controlled directions.
            kd: Damping gain.

        Returns:
            Current arm state.
        """
        self._check_enabled()
        self._check_no_freedrive()
        dynamics = self._get_dynamics()
        kp = self._default_kp if kp is None else kp
        kd = self._default_kd if kd is None else kd
        n = self.num_joints

        q = self._state.joints
        # Latch position on first force_mode call — hold this position in position-controlled directions
        if self._force_mode_origin is None:
            self._force_mode_origin = list(q)
        J = dynamics.jacobian(q)
        wrench_arr = np.asarray(wrench, dtype=float)
        selection_arr = np.asarray(selection, dtype=float)

        # Transform wrench to base frame if task_frame is specified
        if task_frame is not None:
            tf = np.asarray(task_frame, dtype=float)
            R_task = geometry.rotvec_to_matrix(tf[3:6])
            # Rotate force and torque components to base frame
            wrench_arr[:3] = R_task @ wrench_arr[:3]
            wrench_arr[3:] = R_task @ wrench_arr[3:]

        # Force-controlled: Jacobian^T @ wrench → joint torques
        force_torques = J.T @ (selection_arr * wrench_arr)

        # Apply speed limits: scale down force proportionally when approaching limit
        if limits is not None:
            limits_arr = np.asarray(limits, dtype=float)
            v_cart = J @ np.asarray(self._state.velocities, dtype=float)
            scale = 1.0
            for i in range(6):
                if selection_arr[i] > 0 and limits_arr[i] > 0:
                    ratio = abs(v_cart[i]) / limits_arr[i]
                    if ratio > 1.0:
                        scale = 0.0
                        break
                    elif ratio > 0.8:
                        # Linear ramp-down from 80% to 100% of limit
                        s = 1.0 - (ratio - 0.8) / 0.2
                        scale = min(scale, s)
            force_torques *= scale

        # Position-controlled: reduce kp for joints that map to force-controlled directions.
        # Project selection through Jacobian to compute per-joint force-control weight.
        # For each joint, if it primarily contributes to a force-controlled direction,
        # its kp should be reduced so it doesn't fight the torque_ff.
        J_norm = np.abs(J)
        col_norms = J_norm.sum(axis=0)
        col_norms[col_norms < 1e-10] = 1.0
        J_contrib = J_norm / col_norms  # normalized contribution of each joint to each Cartesian DOF
        force_weight = J_contrib.T @ selection_arr  # per-joint: how much it serves force directions
        force_weight = np.clip(force_weight, 0.0, 1.0)
        kps = [(kp * (1.0 - fw)) for fw in force_weight]
        kds = [kd] * n

        g = np.array(self._gravity_torques(list(self._state.joints)))
        self._state = self._backend.command(
            positions=self._force_mode_origin,
            velocities=[0.0] * n,
            kps=kps,
            kds=kds,
            torques=(force_torques + g).tolist(),
        )
        return self._state

    def force_mode_disable(self) -> None:
        """
        Exit force mode, return to normal position control.

        Equivalent to stopping force_mode calls — the arm holds its current position
        with default gains on the next servoj/command cycle.
        """
        self._check_enabled()
        self._force_mode_origin = None
        n = self.num_joints
        g = self._gravity_torques(list(self._state.joints))
        self._state = self._backend.command(
            positions=self._state.joints,
            velocities=[0.0] * n,
            kps=[self._default_kp] * n,
            kds=[self._default_kd] * n,
            torques=g,
        )
        self._safety.reconcile_position(list(self._state.joints))

    # ─── Configuration ────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, path: str, **kwargs) -> "Arm":
        """
        Create an Arm from a YAML config file.

        Config format:
            model: "A1"
            channel: "can0"
            safety:
              max_velocity: 5.0
              timeout_frames: 10
        """
        try:
            import yaml
        except ImportError:
            raise OneroConfigError("pyyaml is required: pip install pyyaml")

        with open(path, 'r') as f:
            raw = yaml.safe_load(f)

        model = raw.get("model", "A1")
        channel = raw.get("channel")
        backend = kwargs.get("backend")

        safety_raw = raw.get("safety", {})
        safety = SafetyConfig(
            max_velocity=safety_raw.get("max_velocity", 5.0),
            max_delta_per_step=safety_raw.get("max_delta_per_step", 0.03),
            timeout_frames=safety_raw.get("timeout_frames", 10),
            enabled=safety_raw.get("enabled", True),
        ) if safety_raw else None

        return cls(model=model, channel=channel, backend=backend,
                   safety=safety, **{k: v for k, v in kwargs.items() if k != "backend"})

    def set_gains(self, kp: float, kd: float) -> None:
        """Update default control gains (also updates GainPolicy fallback)."""
        self._default_kp = kp
        self._default_kd = kd
        self._gain_policy._fallback_kp = kp
        self._gain_policy._fallback_kd = kd

    def set_stiffness(self, stiffness: str) -> None:
        """
        Set global impedance preset: "stiff", "medium", or "soft".

        This controls the adaptive gain policy used by movej/movel/movec.
        Has no effect when explicit kp/kd are passed to motion methods.
        """
        self._gain_policy.stiffness = Stiffness(stiffness)

    def _compute_gains(self, q) -> tuple[list[float], list[float]]:
        """Compute adaptive kp/kd for current configuration. Falls back to fixed gains."""
        try:
            dynamics = self._get_dynamics()
            M = dynamics.mass_matrix(q)
            return self._gain_policy.compute(M)
        except (ImportError, Exception):
            return self._gain_policy.fallback(self.num_joints)

    def get_joint_limits(self) -> list[tuple[float, float]]:
        """
        Get joint position limits.

        Returns:
            List of (min, max) radians per joint.
        """
        return [
            (spec.p_min if spec.p_min is not None else -spec.p_max, spec.p_max)
            for spec in self._model_info.joints
        ]

    def set_payload(self, mass: float, cog: list[float] = None) -> None:
        """
        Configure end-effector payload for accurate dynamics.

        Affects freedrive gravity compensation, collision detection thresholds,
        and external force estimation. Call whenever the tool/payload changes.

        Args:
            mass: Payload mass (kg).
            cog: Center of gravity relative to EE frame [x, y, z] (meters).
                 Default [0, 0, 0] if not specified.
        """
        cog = cog or [0.0, 0.0, 0.0]
        dynamics = self._get_dynamics()
        dynamics.set_payload(mass, cog)
        logger.info(f"Payload set: {mass:.2f} kg, CoG={cog}")

    def get_external_wrench(self) -> np.ndarray:
        """
        Estimate external force/torque at the end-effector.

        Uses torque residual (measured - model predicted) and Jacobian
        to estimate the external wrench. Requires dynamics model.

        Returns:
            Estimated wrench [fx, fy, fz, tx, ty, tz] (N, Nm).
        """
        self._check_enabled()
        dynamics = self._get_dynamics()
        return dynamics.estimate_external_wrench(
            self._state.joints,
            self._state.velocities,
            self._state.torques,
        )

    def check_collision(self) -> bool:
        """
        Check if a collision is occurring based on torque residuals.

        Requires dynamics model and collision_threshold > 0 in SafetyConfig.

        Returns:
            True if collision detected.
        """
        self._check_enabled()
        dynamics = self._get_dynamics()
        residuals = dynamics.torque_residual(
            self._state.joints,
            self._state.velocities,
            self._state.torques,
        )
        return self._safety.check_collision(residuals)

    def calibrate_dynamics(self, num_configs: int = 80,
                           settle_time: float = 1.0,
                           friction_speed: float = 0.3,
                           save_path: Optional[str] = None,
                           configs: Optional[list] = None,
                           seed: Optional[int] = None) -> "CalibrationResult":
        """
        Identify dynamics parameters (inertial + friction) from measurements.

        Phase 1: Static — move to multiple configurations, measure holding
        torques for inertial parameter identification.

        Phase 2: Friction — sweep each joint at constant speed in both
        directions, identify Coulomb and viscous friction per joint.

        After calibration, gravity_torques() and all dynamics methods
        automatically use the identified parameters.

        Args:
            num_configs: Configurations for static phase (60-100 recommended).
            settle_time: Wait time (s) at each static configuration.
            friction_speed: Joint velocity (rad/s) for friction sweeps.
            save_path: If set, save calibration result to this JSON path.
            configs: Manual configuration list (skip automatic generation).
            seed: Random seed for reproducible config generation.

        Returns:
            CalibrationResult with identified parameters and fit statistics.
        """
        self._check_enabled()
        from .calibration import GravityCalibrator, CalibrationResult
        import time as _time

        dynamics = self._get_dynamics()
        calibrator = GravityCalibrator(dynamics._model, dynamics._data)

        # Use physical joint limits from URDF (not motor encoder range)
        joint_limits = list(zip(
            dynamics._model.lowerPositionLimit.tolist(),
            dynamics._model.upperPositionLimit.tolist(),
        ))

        if configs is None:
            configs = calibrator.generate_configs(
                num_configs, joint_limits, margin=0.15, seed=seed
            )

        configs_all, torques_all, vels_all = [], [], []

        # Phase 1: static data
        for i, q in enumerate(configs):
            max_dist = max(
                abs(q[j] - self._state.joints[j]) for j in range(self.num_joints)
            )
            duration = max(1.5, max_dist / 0.8)
            self.movej(q, duration=duration)
            _time.sleep(settle_time)
            state = self._backend.get_state()
            configs_all.append(list(state.joints))
            torques_all.append(list(state.torques))
            vels_all.append([0.0] * self.num_joints)
            if (i + 1) % 10 == 0:
                logger.info(f"Static phase: {i+1}/{len(configs)}")

        # Phase 2: friction sweeps
        dt = 1.0 / self._control_freq
        for j in range(self.num_joints):
            lo, hi = joint_limits[j]
            margin = 0.2 * (hi - lo)
            lo_safe, hi_safe = lo + margin, hi - margin

            start_q = list(self._state.joints)
            start_q[j] = lo_safe
            self.movej(start_q, duration=1.0)
            _time.sleep(0.3)

            steps = int((hi_safe - lo_safe) / (friction_speed * dt))
            for step in range(steps):
                q_cmd = list(self._state.joints)
                q_cmd[j] += friction_speed * dt
                self.servoj(q_cmd)
                if step % 10 == 0:
                    configs_all.append(list(self._state.joints))
                    torques_all.append(list(self._state.torques))
                    vels_all.append(list(self._state.velocities))
            for step in range(steps):
                q_cmd = list(self._state.joints)
                q_cmd[j] -= friction_speed * dt
                self.servoj(q_cmd)
                if step % 10 == 0:
                    configs_all.append(list(self._state.joints))
                    torques_all.append(list(self._state.torques))
                    vels_all.append(list(self._state.velocities))

            logger.info(f"Friction phase: joint {j+1}/{self.num_joints}")

        result = calibrator.fit_with_friction(configs_all, torques_all, vels_all)
        logger.info(
            f"Calibration complete: rank={result.rank}, "
            f"RMS={result.residual_rms:.4f} Nm"
        )
        dynamics.apply_calibration(result.params)

        if save_path:
            result.save(save_path)
            logger.info(f"Calibration saved to {save_path}")

        return result

    def load_calibration(self, path: str) -> None:
        """
        Load and apply a saved calibration result.

        Updates the dynamics model parameters so that gravity_torques()
        and all dependent methods use the calibrated values.

        Args:
            path: Path to calibration JSON file.
        """
        from .calibration import CalibrationResult

        result = CalibrationResult.load(path)
        dynamics = self._get_dynamics()
        dynamics.apply_calibration(result.params)
        logger.info(
            f"Loaded calibration from {path} "
            f"(rank={result.rank}, RMS={result.residual_rms:.4f} Nm)"
        )

    def emergency_stop(self) -> None:
        """
        Immediately disable all motors.

        Call this when an unsafe condition is detected from application code.
        After e-stop, call enable() to resume.
        """
        self._freedrive_active = False
        try:
            self._backend.disable()
        except Exception:
            pass
        self._enabled = False
        self._safety.set_emergency_stop()
        logger.error("EMERGENCY STOP triggered by application")

    # ─── Private ──────────────────────────────────────────────────────────────

    def _on_can_miss(self, joint_idx: int, count: int) -> None:
        """Called by CanBackend when a motor frame is missed."""
        if hasattr(self, '_safety'):
            self._safety.record_response(joint_idx, False)

    def _on_can_fault(self, joint_idx: int, error_code: int) -> None:
        """Called by CanBackend when a motor reports a hardware fault."""
        if hasattr(self, '_safety'):
            self._safety.record_hardware_fault(joint_idx, error_code)

    def _check_enabled(self) -> None:
        if not self._enabled:
            raise OneroError("Arm is not enabled. Call arm.enable() first.")
        if self._safety.emergency_stopped:
            raise OneroError("Arm is in emergency stop. Re-enable to resume.")

    def _check_no_freedrive(self) -> None:
        if self._freedrive_active:
            raise OneroError(
                "Cannot send commands while freedrive is active. "
                "Call freedrive_disable() first."
            )

    def _step_dt(self, dt: float) -> None:
        """Deadline-based sleep: accounts for time spent in command processing."""
        self._clock.tick(dt)

    def _begin_motion(self) -> None:
        """Reset state for a new planned motion."""
        self._clock.reset()
        self._force_mode_origin = None

    @staticmethod
    def _duration_from_speed(distance: float, speed: float,
                             acceleration: float) -> float:
        """Estimate motion duration from trapezoidal speed profile."""
        if distance <= 0 or speed <= 0:
            return 0.5
        t_accel = speed / acceleration
        d_accel = 0.5 * acceleration * t_accel ** 2
        if 2 * d_accel >= distance:
            return 2.0 * math.sqrt(distance / acceleration)
        d_cruise = distance - 2 * d_accel
        t_cruise = d_cruise / speed
        return 2 * t_accel + t_cruise

    def _command_velocity_safe(self, velocities: list[float]) -> ArmState:
        """Send velocity command through safe backend (clamping applied)."""
        return self._safe_backend.command_velocity(velocities)

    def _parse_pose(self, method: str, position, rotation):
        """Parse position/rotation arguments into (pos_3d, rot_matrix)."""
        pos_arr = np.asarray(position, dtype=float)
        if rotation is None:
            if pos_arr.shape == (6,):
                return pos_arr[:3], geometry.rotvec_to_matrix(pos_arr[3:])
            elif pos_arr.shape == (3,):
                raise OneroConfigError(
                    f"{method}: 3D position requires rotation argument. "
                    "Pass rotation separately, or use a 6D pose [x,y,z,rx,ry,rz].")
            else:
                raise OneroConfigError(
                    f"{method}: expected position as [x,y,z] (3D) or "
                    f"[x,y,z,rx,ry,rz] (6D), got shape {pos_arr.shape}")
        else:
            return pos_arr, geometry.ensure_rotation_matrix(rotation)

    def _get_fk(self):
        """Lazy-init FK from URDF."""
        if self._fk is None:
            from .fk import URDFChain
            urdf_path = self._find_urdf()
            if urdf_path is None:
                raise OneroConfigError(
                    "URDF not configured. Set urdf path in config or provide FK manually."
                )
            self._fk = URDFChain(urdf_path)
        return self._fk

    def _get_ik(self):
        """Get IK solver (requires onero_sdk[ik]).

        The solver must maintain internal state for warm-starting: each call
        to solve() should start from the previous solution to avoid jumps.
        """
        try:
            from .ik_solver import PinkIKSolver
        except ImportError:
            raise OneroConfigError(
                "IK requires pinocchio and pink: pip install onero-sdk[ik]"
            )
        if self._ik is None:
            urdf_path = self._find_urdf()
            if urdf_path is None:
                raise OneroConfigError("URDF not configured for IK.")
            self._ik = PinkIKSolver(urdf_path)
            self._ik.reset(np.array(self._state.joints))
        return self._ik

    def _get_dynamics(self):
        """Get dynamics model (requires onero_sdk[dynamics])."""
        if self._dynamics is None:
            try:
                from .dynamics import DynamicsModel
            except ImportError:
                raise OneroConfigError(
                    "Dynamics requires pinocchio: pip install onero-sdk[dynamics]"
                )
            urdf_path = self._find_urdf()
            if urdf_path is None:
                raise OneroConfigError("URDF not configured for dynamics.")
            self._dynamics = DynamicsModel(urdf_path)
        return self._dynamics

    def _gravity_torques(self, joint_positions: list[float]) -> list[float]:
        """Compute gravity compensation torques via dynamics model (pinocchio)."""
        try:
            dynamics = self._get_dynamics()
            return dynamics.gravity_torques(joint_positions)
        except (ImportError, Exception):
            return [0.0] * self.num_joints

    # ------------------------------------------------------------------
    # Public dynamics primitives (require pip install onero-sdk[dynamics])
    # ------------------------------------------------------------------

    def jacobian(self, q: list[float] | None = None) -> np.ndarray:
        """
        End-effector Jacobian (6×n) in world frame.

        Args:
            q: Joint positions (rad). Defaults to current state.

        Returns:
            6×n ndarray. Rows: [vx, vy, vz, wx, wy, wz].
        """
        dynamics = self._get_dynamics()
        if q is None:
            q = self._state.joints
        return dynamics.jacobian(q)

    def gravity_torque(self, q: list[float] | None = None) -> list[float]:
        """
        Gravity compensation torques at given configuration.

        Args:
            q: Joint positions (rad). Defaults to current state.

        Returns:
            Per-joint gravity torques (Nm).
        """
        dynamics = self._get_dynamics()
        if q is None:
            q = self._state.joints
        return dynamics.gravity_torques(q)

    def coriolis_torque(self, q: list[float] | None = None,
                        dq: list[float] | None = None) -> list[float]:
        """
        Coriolis and centrifugal torques at given state.

        Computed as RNEA(q, dq, 0) - RNEA(q, 0, 0), isolating velocity-
        dependent terms from gravity.

        Args:
            q: Joint positions (rad). Defaults to current state.
            dq: Joint velocities (rad/s). Defaults to current state.

        Returns:
            Per-joint Coriolis + centrifugal torques (Nm).
        """
        dynamics = self._get_dynamics()
        if q is None:
            q = self._state.joints
        if dq is None:
            dq = self._state.velocities
        total = np.asarray(dynamics.expected_torques(q, dq))
        gravity = np.asarray(dynamics.gravity_torques(q))
        return (total - gravity).tolist()

    # ------------------------------------------------------------------
    # Kinematic calibration
    # ------------------------------------------------------------------

    def apply_kinematic_calibration(self, result) -> None:
        """
        Apply kinematic calibration corrections to improve absolute accuracy.

        Modifies the internal FK chain in-place. All subsequent FK/IK
        computations will use the corrected parameters.

        Args:
            result: KinematicCalibrationResult from KinematicCalibrator.solve(),
                    or a path to a saved calibration JSON file.
        """
        from .kinematic_calibration import apply_calibration, KinematicCalibrationResult

        if isinstance(result, str):
            result = KinematicCalibrationResult.load(result)

        fk = self._get_fk()
        apply_calibration(fk, result)
        logger.info(
            "Applied kinematic calibration (residual %.2f mm, %d touches)",
            result.residual_mm, result.num_touches,
        )

    def _find_urdf(self) -> Optional[str]:
        """Find URDF path for this model."""
        import os
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        name = self._model_info.name
        candidates = [
            # Packaged URDFs (inside onero_sdk/urdf/ — works after pip install)
            os.path.join(pkg_dir, "urdf", name, f"{name.lower()}_r.urdf"),
            os.path.join(pkg_dir, "urdf", name, f"{name.lower()}_l.urdf"),
            os.path.join(pkg_dir, "urdf", f"{name.lower()}.urdf"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None
