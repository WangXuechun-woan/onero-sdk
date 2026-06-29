"""
Arm — the core primitive of onero SDK.

Provides high-level motion API (movej/movel/movep/servoj/servop) on top of
a pluggable backend. Users interact with this class; the backend handles
whether commands go to real hardware or simulation.
"""

import math
import atexit
import logging
import threading
import weakref
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
from .motion_strategy import (
    MotionContext, cosine_joint_frames, trapezoidal_joint_frames,
    scurve_joint_frames, cartesian_linear_frames, circular_arc_frames,
    velocity_joint_frames, velocity_cartesian_frames,
)
from .compliance import ComplianceController
from .safe_backend import SafeBackend
from .gain_policy import GainPolicy, Stiffness
from .arm_transforms import ArmTransformsMixin
from .mounting import MountingPreset, MountingConfig

logger = logging.getLogger(__name__)


class Arm(ArmTransformsMixin):
    """
    Robotic arm controller (N-DOF, determined by model registry).

    Create with a model name + CAN channel (real hardware):
        arm = Arm("A1R", channel="can0")

    Or with a custom backend (simulation, etc.):
        arm = Arm("A1R", backend=my_backend)

    Or from a config file:
        arm = Arm.from_config("arm.yaml")
    """

    def __init__(self, model: str = "A1R", channel: Optional[str] = None,
                 backend: Optional[ArmBackend] = None,
                 safety: Optional[SafetyConfig] = None,
                 end_effector: Optional[EndEffector] = None,
                 on_error: Optional[Callable] = None,
                 mounting=None,
                 world_frame: bool = False,
                 base_position: Optional[list] = None):
        """
        Args:
            model: Arm model name (e.g. "A1R", "A1L"). Looked up in model registry.
            channel: CAN channel name (e.g. "can0"). Creates a CAN backend.
                     Mutually exclusive with `backend`.
            backend: Custom backend instance. Mutually exclusive with `channel`.
            safety: Safety configuration. Default limits active if None.
            end_effector: Optional gripper or dexterous hand attached to this arm.
            on_error: Callback invoked on critical errors. Signature: (OneroError) -> None.
            mounting: Installation orientation. Accepts None (upright/backward-compat),
                      a MountingPreset enum, a preset string (e.g. "side_mount_left"),
                      or a MountingConfig instance for fine-tuning.
            world_frame: If True, all Cartesian APIs (get_pose, movel, servop, speedl,
                         force_mode, get_external_wrench) operate in world coordinates
                         (X=forward, Y=left, Z=up) instead of base frame.
            base_position: [x, y, z] position of the arm base origin in world
                           coordinates (meters). Required for multi-arm setups
                           sharing a world frame. Defaults to [0, 0, 0].
        """
        self._model_info = get_model(model)
        self._on_error = on_error
        self._enabled = False

        # Mounting, world-frame, and TCP transforms
        self._init_transforms(mounting, world_frame, base_position)

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
            if spec.p_min > spec.p_max:
                raise OneroConfigError(
                    f"Joint {i}: invalid limits p_min={spec.p_min} > p_max={spec.p_max}")
            joint_limits.append((spec.p_min, spec.p_max))
        self._safety = SafetyMonitor(
            num_joints=self._model_info.num_joints,
            joint_limits=joint_limits,
            config=safety or SafetyConfig(),
            on_error=on_error,
            on_stop=lambda: self._backend.disable(),
        )

        # Wrap backend with safety layer (transparent clamping)
        torque_limits = [spec.t_max for spec in self._model_info.joints]
        self._safe_backend = SafeBackend(self._backend, self._safety,
                                         torque_limits=torque_limits)

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
        self._motion_lock = threading.Lock()  # prevents concurrent motion commands
        self._clock = ControlClock()

        # Motion executor (unified control loop — uses safe backend)
        self._executor = MotionExecutor(
            backend=self._safe_backend,
            safety=self._safety,
            recorder=self._recorder,
            clock=self._clock,
            get_state=lambda: self._state,
            set_state=self._set_state_locked,
            gravity_torques_fn=self._gravity_torques,
        )

        # Force mode state
        self._force_mode_origin: Optional[list[float]] = None
        self._compliance: Optional[ComplianceController] = None

        # Control parameters from model
        self._default_kp = self._model_info.default_kp
        self._default_kd = self._model_info.default_kd
        self._control_freq = self._model_info.control_freq

        # Reflected motor inertia: I_rotor × gear_ratio² per joint
        self._reflected_inertia = None
        if self._model_info.gear_ratios and self._model_info.rotor_inertias:
            import numpy as np
            ratios = np.array(self._model_info.gear_ratios)
            rotors = np.array(self._model_info.rotor_inertias)
            self._reflected_inertia = rotors * (ratios ** 2)

        self._gain_policy = GainPolicy(
            stiffness=Stiffness.STIFF,
            fallback_kp=self._default_kp,
            fallback_kd=self._default_kd,
            dynamics_fn=self._get_dynamics,
            reflected_inertia=self._reflected_inertia,
            cache_interval=10,
        )

        # Safety: auto-disable on interpreter exit if user forgets
        self._atexit_registered = True
        weakself = weakref.ref(self)
        def _cleanup():
            arm = weakself()
            if arm is not None and arm._enabled:
                logger.warning(
                    "Arm was not disabled before exit. "
                    "Calling disable() automatically. "
                    "Use 'with Arm(...) as arm:' or try/finally to avoid this.")
                try:
                    arm.disable()
                except Exception:
                    pass
        atexit.register(_cleanup)

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
        with self._lock:
            return self._state

    @property
    def joints(self) -> list[float]:
        """Current joint positions (rad). Shortcut for state.joints."""
        with self._lock:
            return list(self._state.joints)

    @property
    def is_moving(self) -> bool:
        """True if any joint velocity exceeds 0.01 rad/s."""
        with self._lock:
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
        state = self._backend.enable()
        with self._lock:
            self._state = state
        self._enabled = True
        self._safety.reset(state.joints)
        if self._end_effector is not None:
            self._end_effector.enable()
        logger.info(f"Arm enabled. Initial joints: {state.joints}")
        return state

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
        state = self._backend.get_state()
        with self._lock:
            self._state = state
        return state

    def get_pose(self, frame: Optional[str] = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute current end-effector pose via FK.

        If set_tcp() has been called, returns TCP pose (not flange).

        Args:
            frame: Coordinate frame for the returned pose.
                   "base" = arm base frame, "world" = world frame,
                   "tool" = current tool frame, or a custom registered name.
                   None = use default (from constructor's world_frame param).

        Returns:
            (position [3], rotation_matrix [3x3])

        Raises:
            OneroConfigError: if URDF not configured for this model.
        """
        fk = self._get_fk()
        with self._lock:
            joints = list(self._state.joints)
        pos, rot = fk.get_ee_pose(joints)
        return self._output_pose_transform(pos, rot, frame=frame)

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
        try:
            use_adaptive = (kp is None and kd is None)
            if not use_adaptive:
                kp = self._default_kp if kp is None else kp
                kd = self._default_kd if kd is None else kd
            n = self.num_joints

            joints = self._validate_joints(joints)

            with self._lock:
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

            ctx = MotionContext(
                start_joints=start_pos, num_joints=n, dt=dt, steps=steps,
                compute_gains=self._compute_gains if use_adaptive else None,
                fixed_kp=kp if not use_adaptive else None,
                fixed_kd=kd if not use_adaptive else None,
            )

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
                    s_dur = max(p.duration for p in profiles)
                    actual_dur = max(actual_dur, s_dur)
                    actual_steps = max(1, int(actual_dur / dt))
                    _frames = lambda: scurve_joint_frames(ctx, profiles, actual_steps)
                else:
                    joint_profiles = multi_joint_trapezoidal(
                        start_pos, joints, v_maxs, a_maxs, dt,
                        target_duration=actual_dur)
                    _frames = lambda: trapezoidal_joint_frames(ctx, joint_profiles)
            else:
                _frames = lambda: cosine_joint_frames(ctx, joints, duration)

            self._state = self._executor.run(_frames())

            if not self._safety.emergency_stopped:
                max_err = max(abs(self._state.joints[i] - joints[i]) for i in range(n))
                if max_err > 0.01:
                    logger.warning(
                        f"movej: target clamped by safety limits (max error: {max_err:.3f} rad)")

            return self._state
        finally:
            self._end_motion()

    def movel(self, position: Union[list, np.ndarray],
              rotation: Union[list, np.ndarray] = None,
              duration: Optional[float] = None,
              speed: Optional[float] = None, acceleration: Optional[float] = None,
              kp: Optional[float] = None, kd: Optional[float] = None,
              frame: Optional[str] = None) -> ArmState:
        """
        Move end-effector along a straight line in Cartesian space.

        Interpolates linearly in position and uses SLERP for rotation.
        Solves IK at each step.

        Args:
            position: Target EE position [x, y, z] (meters), or a 6D pose
                      [x, y, z, rx, ry, rz] (meters + rotation vector) when
                      rotation is not provided separately.
                      When frame="tool", this is a DISPLACEMENT relative to the
                      current TCP pose in tool-frame axes (not an absolute target).
            rotation: Target EE rotation (3x3 matrix, quaternion [x,y,z,w],
                      or rotvec [rx,ry,rz]). Optional if position is 6D.
                      When frame="tool", this is a relative rotation applied
                      in the tool frame (post-multiplied).
            duration: Motion duration (seconds). Auto-computed if None.
            speed: Max tool speed (m/s). Alternative to duration —
                   duration is computed from the Cartesian distance.
                   Ignored if duration is explicitly set.
            acceleration: Tool acceleration (m/s²). Used with speed to
                         compute duration. Default 1.2 m/s² (UR-compatible).
            kp: Position gain.
            kd: Damping gain.
            frame: Coordinate frame for target pose.
                   "base"/"world"/"tool"/custom name. None = default frame.

        Returns:
            Final arm state.
        """
        self._check_enabled()
        self._check_no_freedrive()
        self._begin_motion()
        try:
            ik = self._get_ik()
            use_adaptive = (kp is None and kd is None)
            if not use_adaptive:
                kp = self._default_kp if kp is None else kp
                kd = self._default_kd if kd is None else kd
            n = self.num_joints

            # Parse pose: accept [x,y,z,rx,ry,rz] or separate position+rotation
            target_pos, target_rot = self._parse_pose("movel", position, rotation, frame=frame)

            # Get current EE pose in base/flange frame (same frame as _parse_pose output)
            fk = self._get_fk()
            with self._lock:
                current_joints = list(self._state.joints)
            start_pos_ee, start_rot_ee = fk.get_ee_pose(current_joints)

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

            ctx = MotionContext(
                start_joints=current_joints, num_joints=n, dt=dt, steps=steps,
                compute_gains=self._compute_gains if use_adaptive else None,
                fixed_kp=kp if not use_adaptive else None,
                fixed_kd=kd if not use_adaptive else None,
            )

            self._state = self._executor.run(
                cartesian_linear_frames(ctx, start_pos_ee, start_rot_ee,
                                        target_pos, target_rot, ik,
                                        geometry.slerp_matrix)
            )

            return self._state
        finally:
            self._end_motion()

    def movep(self, position: Union[list, np.ndarray],
              rotation: Union[list, np.ndarray] = None,
              duration: Optional[float] = None,
              speed: Optional[float] = None, acceleration: Optional[float] = None,
              kp: Optional[float] = None, kd: Optional[float] = None,
              frame: Optional[str] = None) -> ArmState:
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
            frame: Coordinate frame for target pose.
                   "base"/"world"/"tool"/custom name. None = default frame.

        Returns:
            Final arm state.
        """
        self._check_enabled()
        self._check_no_freedrive()
        ik = self._get_ik()

        target_pos, target_rot = self._parse_pose("movep", position, rotation, frame=frame)

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
              kp: Optional[float] = None, kd: Optional[float] = None,
              frame: Optional[str] = None) -> ArmState:
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
            frame: Coordinate frame for via/target poses.
                   "base"/"world"/"tool"/custom name. None = default frame.

        Returns:
            Final arm state.
        """
        self._check_enabled()
        self._check_no_freedrive()
        self._begin_motion()
        try:
            ik = self._get_ik()
            fk = self._get_fk()
            use_adaptive = (kp is None and kd is None)
            if not use_adaptive:
                kp = self._default_kp if kp is None else kp
                kd = self._default_kd if kd is None else kd
            n = self.num_joints

            # Parse via and target through coordinate transforms
            p1, r1 = self._parse_pose("movec", via_pose, None, frame=frame)
            p2, r2 = self._parse_pose("movec", target_pose, None, frame=frame)

            # Start point: current FK in base/flange frame (same as _parse_pose output)
            with self._lock:
                current_joints = list(self._state.joints)
            p0, r0 = fk.get_ee_pose(current_joints)

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

            ctx = MotionContext(
                start_joints=current_joints, num_joints=n, dt=dt, steps=steps,
                compute_gains=self._compute_gains if use_adaptive else None,
                fixed_kp=kp if not use_adaptive else None,
                fixed_kd=kd if not use_adaptive else None,
            )

            self._state = self._executor.run(
                circular_arc_frames(ctx, center, v0, normal, total_angle,
                                    r0, r1, r2, ik, geometry.slerp_matrix,
                                    geometry.rotate_vector)
            )
            return self._state
        finally:
            self._end_motion()

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
               tau_ff: Optional[list[float]] = None,
               velocities: Optional[list[float]] = None) -> ArmState:
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
            velocities: Velocity feedforward per joint (rad/s). Improves tracking
                        during fast motion. If None, zeros are sent.

        Returns:
            Current arm state from feedback.
        """
        self._check_enabled()
        self._check_no_freedrive()
        kp = self._default_kp if kp is None else kp
        kd = self._default_kd if kd is None else kd
        n = self.num_joints

        joints = self._validate_joints(joints)

        if tau_ff is not None:
            torques = list(tau_ff)
        else:
            with self._lock:
                cur_joints = list(self._state.joints)
            torques = self._gravity_torques(cur_joints)

        vel_cmd = list(velocities) if velocities is not None else [0.0] * n

        state = self._safe_backend.command(
            positions=joints,
            velocities=vel_cmd,
            kps=[kp] * n,
            kds=[kd] * n,
            torques=torques,
        )
        with self._lock:
            self._state = state
        self._recorder.record_frame(
            state.joints, state.velocities, state.torques)
        return state

    def servop(self, position: Union[list, np.ndarray],
               rotation: Union[list, np.ndarray] = None,
               kp: Optional[float] = None, kd: Optional[float] = None,
               frame: Optional[str] = None) -> ArmState:
        """
        Send a single task-space target (real-time control).

        Solves IK incrementally then calls servoj. Call at control frequency.

        Args:
            position: Target EE position [x, y, z] (meters), or a 6D pose
                      [x, y, z, rx, ry, rz] when rotation is not provided.
                      When frame="tool", this is a DISPLACEMENT relative to the
                      current TCP pose in tool-frame axes.
            rotation: Target EE rotation (3x3 matrix, quaternion [x,y,z,w],
                      or rotvec [rx,ry,rz]). Optional if position is 6D.
                      When frame="tool", this is a relative rotation in tool frame.
            kp: Position gain.
            kd: Damping gain.
            frame: Coordinate frame for target pose.
                   "base"/"world"/"tool"/custom name. None = default frame.

        Returns:
            Current arm state from feedback.
        """
        self._check_enabled()
        ik = self._get_ik()

        target_pos, target_rot = self._parse_pose("servop", position, rotation, frame=frame)

        q = ik.solve(target_pos, target_rot, dt=1.0 / self._control_freq)
        joints = q.tolist() if hasattr(q, 'tolist') else list(q)

        return self.servoj(joints, kp=kp, kd=kd)

    servol = servop

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
        try:
            n = self.num_joints
            dt = 1.0 / self._control_freq

            velocities = self._validate_joints(velocities, param_name="velocities")

            target_vel = list(velocities)

            max_vel_delta = max(abs(v) for v in target_vel) if target_vel else 0
            ramp_time = max_vel_delta / acceleration if acceleration > 0 else 0
            ramp_steps = int(ramp_time * self._control_freq)

            if duration is not None:
                hold_time = max(0, duration - 2 * ramp_time)
            else:
                hold_time = 1.0
            hold_steps = int(hold_time * self._control_freq)

            self._state = self._executor.run(
                velocity_joint_frames(target_vel, ramp_steps, hold_steps, dt)
            )

            # Gradual transition to position hold (ramp kp over ~50ms to avoid torque jump)
            settle_steps = max(1, int(0.05 * self._control_freq))
            cur_joints = list(self._state.joints)
            g = self._gravity_torques(cur_joints)
            for step in range(settle_steps):
                alpha = (step + 1) / settle_steps
                kps = [alpha * self._default_kp] * n
                kds = [self._default_kd] * n
                self._state = self._safe_backend.command(
                    positions=cur_joints,
                    velocities=[0.0] * n,
                    kps=kps, kds=kds, torques=g,
                    hold=True,
                )
                self._clock.tick(dt)
            self._safety.reconcile_position(list(self._state.joints))
            return self._state
        finally:
            self._end_motion()

    def speedl(self, velocities: list[float], acceleration: float = 0.5,
               duration: Optional[float] = None,
               frame: Optional[str] = None) -> ArmState:
        """
        Move end-effector at target Cartesian velocity.

        Recomputes Jacobian each frame to maintain accurate Cartesian velocity
        as the arm configuration changes.

        Args:
            velocities: Target EE velocity [vx, vy, vz, wx, wy, wz] (m/s, rad/s).
                        When frame="tool", velocities are in tool-frame axes
                        (e.g. vz = along tool Z axis).
            acceleration: Joint acceleration limit (rad/s^2).
            duration: Total motion time (seconds). Default 1s hold after ramp-up.
            frame: Coordinate frame for the velocity vector.
                   "base"/"world"/"tool"/custom name. None = default frame.

        Returns:
            Final arm state.
        """
        self._check_enabled()
        self._check_no_freedrive()
        self._begin_motion()
        try:
            dynamics = self._get_dynamics()
            twist = np.asarray(velocities, dtype=float)
            if twist.shape != (6,):
                raise OneroConfigError(
                    f"speedl: expected 6-element twist [vx,vy,vz,wx,wy,wz], got shape {twist.shape}")

            resolved_frame = self._resolve_frame(frame)
            is_dynamic_frame = (resolved_frame == "tool")

            # For dynamic frames (tool), re-transform each tick
            if is_dynamic_frame:
                twist_fn = lambda: self._input_twist_transform(twist, frame=frame)
                twist_base = twist_fn()
            else:
                twist_base = self._input_twist_transform(twist, frame=frame)
                twist_fn = None

            n = self.num_joints
            dt = 1.0 / self._control_freq

            # Compute initial joint velocity for ramp timing
            joint_vel_target = dynamics.jacobian_solve(self._state.joints, twist_base)
            max_vel_delta = float(np.max(np.abs(joint_vel_target)))
            ramp_time = max_vel_delta / acceleration if acceleration > 0 else 0
            ramp_steps = int(ramp_time * self._control_freq)

            if duration is not None:
                hold_time = max(0, duration - 2 * ramp_time)
            else:
                hold_time = 1.0
            hold_steps = int(hold_time * self._control_freq)

            def _get_joints():
                return self._state.joints

            self._state = self._executor.run(
                velocity_cartesian_frames(twist_base, ramp_steps, hold_steps, dt,
                                          _get_joints, dynamics.jacobian_solve,
                                          twist_fn=twist_fn)
            )

            # Gradual transition to position hold
            settle_steps = max(1, int(0.05 * self._control_freq))
            cur_joints = list(self._state.joints)
            g = self._gravity_torques(cur_joints)
            for step in range(settle_steps):
                alpha = (step + 1) / settle_steps
                kps = [alpha * self._default_kp] * n
                kds = [self._default_kd] * n
                self._state = self._safe_backend.command(
                    positions=cur_joints,
                    velocities=[0.0] * n,
                    kps=kps, kds=kds, torques=g,
                    hold=True,
                )
                self._clock.tick(dt)
            self._safety.reconcile_position(list(self._state.joints))
            return self._state
        finally:
            self._end_motion()

    def stop(self, deceleration: float = 2.0) -> ArmState:
        """
        Decelerate to zero velocity and hold position.

        Can be called while another motion is running (from a different thread)
        to interrupt it. Triggers emergency stop to abort the in-progress motion,
        then performs controlled deceleration.

        Args:
            deceleration: Rate of deceleration (rad/s^2).

        Returns:
            Final arm state.
        """
        self._check_enabled()
        self._check_no_freedrive()

        # If another motion holds the lock, signal it to stop
        if not self._motion_lock.acquire(blocking=False):
            self._safety.set_emergency_stop()
            self._motion_lock.acquire(blocking=True)
            self._safety.clear_emergency_stop()

        try:
            self._clock.reset()
            self._force_mode_origin = None
            n = self.num_joints
            dt = 1.0 / self._control_freq

            # Get fresh state for accurate deceleration
            self._state = self._backend.get_state()
            current_vel = list(self._state.velocities)
            max_vel = max(abs(v) for v in current_vel) if current_vel else 0

            if max_vel < 1e-6:
                g = self._gravity_torques(list(self._state.joints))
                self._state = self._safe_backend.command(
                    positions=list(self._state.joints),
                    velocities=[0.0] * n,
                    kps=[self._default_kp] * n,
                    kds=[self._default_kd] * n,
                    torques=g,
                    hold=True,
                )
                return self._state

            ramp_steps = int((max_vel / deceleration) * self._control_freq)

            def _frames():
                for step in range(ramp_steps):
                    alpha = 1.0 - (step + 1) / max(ramp_steps, 1)
                    yield VelocityFrame(velocities=[alpha * v for v in current_vel], dt=dt)

            self._state = self._executor.run(_frames())

            self._state = self._command_velocity_safe([0.0] * n)
            g = self._gravity_torques(list(self._state.joints))
            self._state = self._safe_backend.command(
                positions=list(self._state.joints),
                velocities=[0.0] * n,
                kps=[self._default_kp] * n,
                kds=[self._default_kd] * n,
                torques=g,
                hold=True,
            )
            self._safety.reconcile_position(list(self._state.joints))
            return self._state
        finally:
            self._end_motion()

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
        try:
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
        finally:
            self._end_motion()

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
        try:
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
        finally:
            self._end_motion()

    # ─── Freedrive (Teach Mode) ──────────────────────────────────────────────

    def freedrive_enable(self, damping: float = 0.5,
                         free_axes: Optional[list[int]] = None,
                         lock_stiffness: float = 200.0) -> None:
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
            lock_stiffness: Position gain for locked axes (N/m or Nm/rad).
                           Higher = stiffer constraint. Default 200.0.
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
            pos_lock = None
            rot_lock = None
            compliance = ComplianceController(dynamics, n) if lock_mask is not None else None
            try:
                while self._freedrive_active:
                    if self._safety.emergency_stopped:
                        break
                    with self._lock:
                        q = list(self._state.joints)

                    gravity_torques = dynamics.gravity_torques(q)
                    torques = list(gravity_torques)

                    if lock_mask is not None:
                        if q_lock is None:
                            q_lock = np.array(q)
                            pos_lock, rot_lock = dynamics.ee_pose(q)
                        J = dynamics.jacobian(q)
                        tau_lock = compliance.freedrive_lock_torques(
                            q, lock_mask, pos_lock, rot_lock, J, lock_stiffness)
                        torques = (np.array(torques) + tau_lock).tolist()

                    state = self._safe_backend.command(
                        positions=q,
                        velocities=[0.0] * n,
                        kps=[0.0] * n,
                        kds=[damping] * n,
                        torques=torques,
                        hold=True,
                    )
                    with self._lock:
                        self._state = state
                    self._recorder.record_frame(
                        state.joints, state.velocities, state.torques)
                    clock.tick(dt)
            except Exception as e:
                logger.error(f"Freedrive loop crashed: {e}. Triggering emergency stop.")
                self._safety._trigger_emergency_stop(
                    0, f"Freedrive loop exception: {e}")
                self._freedrive_active = False

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
        with self._lock:
            cur_joints = list(self._state.joints)
        g = self._gravity_torques(cur_joints)
        state = self._safe_backend.command(
            positions=cur_joints,
            velocities=[0.0] * n,
            kps=[self._default_kp] * n,
            kds=[self._default_kd] * n,
            torques=g,
            hold=True,
        )
        with self._lock:
            self._state = state
        self._safety.reconcile_position(cur_joints)
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

        self._state = self._safe_backend.command(
            positions=q,
            velocities=[0.0] * n,
            kps=[0.0] * n,
            kds=[damping] * n,
            torques=gravity_torques,
            hold=True,
        )
        self._recorder.record_frame(
            self._state.joints, self._state.velocities, self._state.torques)
        return self._state

    # ─── Force Mode ──────────────────────────────────────────────────────────

    def force_mode(self, selection: list[int], wrench: list[float],
                   task_frame: Optional[list[float]] = None,
                   limits: Optional[list[float]] = None,
                   kp: Optional[float] = None, kd: Optional[float] = None,
                   frame: Optional[str] = None) -> ArmState:
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
            frame: Coordinate frame for the wrench vector.
                   "base"/"world"/"tool"/custom name. None = default frame.
                   When task_frame is explicitly provided, frame= is ignored
                   and a warning is logged if both are set.

        Returns:
            Current arm state.
        """
        self._check_enabled()
        self._check_no_freedrive()
        dynamics = self._get_dynamics()
        kp = self._default_kp if kp is None else kp
        kd = self._default_kd if kd is None else kd
        n = self.num_joints

        # Validate inputs before any computation
        sel_arr = np.asarray(selection, dtype=float)
        if sel_arr.shape != (6,):
            raise OneroConfigError(
                f"force_mode: selection must be 6-element, got shape {sel_arr.shape}")
        if not np.all(np.isfinite(sel_arr)):
            raise OneroConfigError("force_mode: selection contains NaN or Inf")
        if not np.all((sel_arr == 0) | (sel_arr == 1)):
            raise OneroConfigError(
                "force_mode: selection values must be 0 or 1")

        wrench_arr = np.asarray(wrench, dtype=float)
        if wrench_arr.shape != (6,):
            raise OneroConfigError(
                f"force_mode: wrench must be 6-element, got shape {wrench_arr.shape}")
        if not np.all(np.isfinite(wrench_arr)):
            raise OneroConfigError("force_mode: wrench contains NaN or Inf")

        if limits is not None:
            limits_arr = np.asarray(limits, dtype=float)
            if limits_arr.shape != (6,):
                raise OneroConfigError(
                    f"force_mode: limits must be 6-element, got shape {limits_arr.shape}")
            if not np.all(np.isfinite(limits_arr)):
                raise OneroConfigError("force_mode: limits contains NaN or Inf")
            if np.any(limits_arr < 0):
                raise OneroConfigError("force_mode: limits must be non-negative")

        if not np.isfinite(kp) or kp < 0:
            raise OneroConfigError(f"force_mode: kp must be finite and non-negative, got {kp}")
        if not np.isfinite(kd) or kd < 0:
            raise OneroConfigError(f"force_mode: kd must be finite and non-negative, got {kd}")

        with self._lock:
            q = self._state.joints
            v_joints = self._state.velocities
        if self._force_mode_origin is None:
            self._force_mode_origin = list(q)
        J = dynamics.jacobian(q)

        task_frame_rotation = None
        if task_frame is not None:
            if frame is not None:
                logger.warning(
                    "force_mode: both task_frame and frame= provided; "
                    "task_frame takes precedence, frame= is ignored.")
            tf = np.asarray(task_frame, dtype=float)
            if tf.shape != (6,):
                raise OneroConfigError(
                    f"force_mode: task_frame must be 6-element [x,y,z,rx,ry,rz], "
                    f"got shape {tf.shape}")
            if not np.all(np.isfinite(tf)):
                raise OneroConfigError("force_mode: task_frame contains NaN or Inf")
            task_frame_rotation = geometry.rotvec_to_matrix(tf[3:6])
        elif frame is not None or self._default_frame != "base":
            task_frame_rotation = self._resolve_frame_rotation(frame)

        if self._compliance is None:
            self._compliance = ComplianceController(dynamics, n)

        # TCP moment arm: if Jacobian is at flange but wrench is specified at TCP,
        # we need to add the r × f coupling to torque components.
        tcp_offset_base = None
        if self._tcp_configured:
            fk = self._get_fk()
            flange_pos, flange_rot = fk.get_ee_pose(list(q))
            tcp_offset_base = flange_rot @ self._tcp_offset_pos

        kps, kds, force_torques = self._compliance.force_step(
            q, v_joints, sel_arr, wrench_arr, J,
            task_frame_rotation=task_frame_rotation,
            tcp_offset_base=tcp_offset_base,
            limits=limits, kp=kp, kd=kd,
        )

        g = np.array(dynamics.gravity_torques(list(q)))
        state = self._safe_backend.command(
            positions=self._force_mode_origin,
            velocities=[0.0] * n,
            kps=kps,
            kds=kds,
            torques=(force_torques + g).tolist(),
            hold=True,
        )
        with self._lock:
            self._state = state
        return state

    def force_mode_disable(self) -> None:
        """
        Exit force mode, return to normal position control.

        Equivalent to stopping force_mode calls — the arm holds its current position
        with default gains on the next servoj/command cycle.
        """
        self._check_enabled()
        self._force_mode_origin = None
        n = self.num_joints
        with self._lock:
            cur_joints = list(self._state.joints)
        g = self._gravity_torques(cur_joints)
        state = self._safe_backend.command(
            positions=cur_joints,
            velocities=[0.0] * n,
            kps=[self._default_kp] * n,
            kds=[self._default_kd] * n,
            torques=g,
            hold=True,
        )
        with self._lock:
            self._state = state
        self._safety.reconcile_position(list(state.joints))

    # ─── Configuration ────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, path: str, **kwargs) -> "Arm":
        """
        Create an Arm from a YAML config file.

        Config format:
            model: "A1R"
            channel: "can0"
            mounting: "side_mount_right"
            world_frame: true
            base_position: [0, -0.15, 0]
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

        model = raw.get("model", "A1R")
        channel = raw.get("channel")
        backend = kwargs.get("backend")
        mounting = raw.get("mounting")
        world_frame = raw.get("world_frame", False)
        base_position = raw.get("base_position")

        safety_raw = raw.get("safety", {})
        safety = SafetyConfig(
            max_velocity=safety_raw.get("max_velocity", 5.0),
            max_delta_per_step=safety_raw.get("max_delta_per_step", 0.03),
            timeout_frames=safety_raw.get("timeout_frames", 10),
            enabled=safety_raw.get("enabled", True),
        ) if safety_raw else None

        extra = {k: v for k, v in kwargs.items()
                 if k not in ("backend", "mounting", "world_frame", "base_position")}
        return cls(model=model, channel=channel, backend=backend,
                   mounting=mounting, world_frame=world_frame,
                   base_position=base_position, safety=safety, **extra)

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
        return self._gain_policy.compute_adaptive(q, self.num_joints)

    def get_joint_limits(self) -> list[tuple[float, float]]:
        """
        Get joint position limits.

        Returns:
            List of (min, max) radians per joint.
        """
        return [
            (spec.p_min, spec.p_max)
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

    def estimate_payload(self, num_samples: int = 50,
                         settle_time: float = 0.5) -> float:
        """
        Estimate payload mass from torque residual at current static configuration.

        The arm must be stationary (holding position). Compares measured torques
        against model-predicted gravity torques (without payload) and solves for
        the mass that explains the residual via the Jacobian.

        After estimation, automatically calls set_payload() to update dynamics.

        Args:
            num_samples: Number of torque readings to average (reduces noise).
            settle_time: Seconds to wait for arm to settle before measuring.

        Returns:
            Estimated payload mass (kg). Returns 0 if estimation fails.
        """
        import time

        self._check_enabled()
        dynamics = self._get_dynamics()

        # Wait for the arm to be fully static
        time.sleep(settle_time)

        # Average multiple torque readings to reduce noise
        tau_samples = []
        q_samples = []
        dt = 1.0 / self._control_freq
        for _ in range(num_samples):
            state = self._backend.get_state()
            tau_samples.append(np.array(state.torques))
            q_samples.append(np.array(state.joints))
            time.sleep(dt)

        tau_measured = np.mean(tau_samples, axis=0)
        q = np.mean(q_samples, axis=0)

        # Compute baseline gravity torques WITHOUT payload (thread-safe, no mutation)
        tau_expected = np.array(dynamics.gravity_torques_no_payload(q.tolist()))

        # Residual = what the payload contributes
        tau_residual = tau_measured - tau_expected

        # Get gravity direction in base frame
        g_vec = dynamics.get_gravity_vector()

        # Jacobian: translational part maps EE force → joint torques
        J = dynamics.jacobian(q.tolist())
        J_linear = J[:3, :]  # 3×n

        # τ_residual ≈ J_linear.T @ (m * g_vec)
        # Solve least-squares for m: J_linear.T @ g_vec * m = τ_residual
        g_torque_unit = J_linear.T @ g_vec  # n×1: torque per unit mass
        norm_sq = np.dot(g_torque_unit, g_torque_unit)

        if norm_sq < 1e-10:
            logger.warning("Payload estimation failed: Jacobian×gravity near zero "
                           "(arm may be in singular configuration)")
            return 0.0

        mass = np.dot(g_torque_unit, tau_residual) / norm_sq
        mass = max(0.0, mass)  # Mass can't be negative

        # Apply payload (assume CoG at EE origin — good enough for most grippers)
        if mass > 0.01:
            self.set_payload(mass, [0.0, 0.0, 0.0])
            logger.info(f"Payload estimated: {mass:.3f} kg")
        else:
            logger.info("No significant payload detected")

        return mass

    def enable_auto_payload(self, velocity_threshold: float = 0.02,
                            quiet_duration: float = 0.3,
                            min_change: float = 0.05) -> None:
        """
        Enable automatic payload estimation at motion boundaries.

        At the start of each planned motion (movej/movel/movec/speedj/speedl),
        checks whether the arm is quasi-static. If all joint velocities are below
        threshold for quiet_duration, re-estimates payload mass from torque residuals.
        Updates dynamics only if estimated mass differs from current by more than
        min_change kg.

        Estimation triggers at motion boundaries, not continuously mid-trajectory.
        This avoids compensation discontinuities during execution while still
        adapting to grasped objects between motions.

        Args:
            velocity_threshold: Max joint velocity (rad/s) to count as "static".
            quiet_duration: Seconds of continuous quiet needed to trigger estimation.
            min_change: Minimum mass change (kg) to update dynamics (avoids jitter).
        """
        self._check_enabled()
        self._get_dynamics()  # Verify dynamics available

        self._auto_payload_config = {
            "velocity_threshold": velocity_threshold,
            "quiet_duration": quiet_duration,
            "min_change": min_change,
        }
        self._auto_payload_mass = 0.0
        self._auto_payload_quiet_start = None
        self._auto_payload_enabled = True
        logger.info(f"Auto payload enabled (threshold={velocity_threshold} rad/s, "
                    f"quiet={quiet_duration}s)")

    def disable_auto_payload(self) -> None:
        """Disable automatic payload estimation."""
        self._auto_payload_enabled = False
        logger.info("Auto payload disabled")

    def _check_auto_payload(self) -> None:
        """Called from control loop — check if we should re-estimate payload."""
        if not getattr(self, '_auto_payload_enabled', False):
            return

        import time
        config = self._auto_payload_config
        state = self._state
        if state is None:
            return

        # Check if all joints are quasi-static
        max_vel = max(abs(v) for v in state.velocities)
        now = time.monotonic()

        if max_vel < config["velocity_threshold"]:
            if self._auto_payload_quiet_start is None:
                self._auto_payload_quiet_start = now
            elif now - self._auto_payload_quiet_start >= config["quiet_duration"]:
                # Quiet long enough → estimate
                self._auto_payload_quiet_start = None
                self._do_quick_payload_estimate(state, config["min_change"])
        else:
            self._auto_payload_quiet_start = None

    def _do_quick_payload_estimate(self, state, min_change: float) -> None:
        """Quick single-sample payload estimate (called from control loop context)."""
        try:
            dynamics = self._get_dynamics()
            q = np.array(state.joints)
            tau_measured = np.array(state.torques)

            old_mass = getattr(self, '_auto_payload_mass', 0.0)

            # Thread-safe: compute baseline without mutating shared state
            tau_expected = np.array(dynamics.gravity_torques_no_payload(q.tolist()))

            tau_residual = tau_measured - tau_expected

            g_vec = dynamics.get_gravity_vector()

            J = dynamics.jacobian(q.tolist())
            J_linear = J[:3, :]
            g_torque_unit = J_linear.T @ g_vec
            norm_sq = np.dot(g_torque_unit, g_torque_unit)

            if norm_sq < 1e-10:
                return

            mass = max(0.0, np.dot(g_torque_unit, tau_residual) / norm_sq)

            if abs(mass - old_mass) > min_change:
                self._auto_payload_mass = mass
                if mass > 0.01:
                    dynamics.set_payload(mass, [0.0, 0.0, 0.0])
                logger.info(f"Auto payload updated: {old_mass:.3f} → {mass:.3f} kg")
        except Exception as e:
            logger.debug(f"Auto payload estimate failed: {e}")

    def get_external_wrench(self, frame: Optional[str] = None) -> np.ndarray:
        """
        Estimate external force/torque at the end-effector.

        Uses torque residual (measured - model predicted) and Jacobian
        to estimate the external wrench. Requires dynamics model.

        Args:
            frame: Coordinate frame for the returned wrench.
                   "base"/"world"/"tool"/custom name. None = default frame.

        Returns:
            Estimated wrench [fx, fy, fz, tx, ty, tz] (N, Nm).
        """
        self._check_enabled()
        dynamics = self._get_dynamics()
        wrench = dynamics.estimate_external_wrench(
            self._state.joints,
            self._state.velocities,
            self._state.torques,
        )
        return self._output_wrench_transform(wrench, frame=frame)

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
        # Record the gravity direction used during calibration
        result.gravity_vector = self._gravity_in_base.tolist()
        logger.info(
            f"Calibration complete: rank={result.rank}, "
            f"RMS={result.residual_rms:.4f} Nm, "
            f"gravity={result.gravity_vector}"
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

        Warns if the calibration was performed with a different gravity
        direction (mounting orientation) than the current configuration.

        Args:
            path: Path to calibration JSON file.
        """
        from .calibration import CalibrationResult

        result = CalibrationResult.load(path)

        # Check gravity direction consistency
        if result.gravity_vector is not None:
            cal_g = np.asarray(result.gravity_vector)
            cur_g = self._gravity_in_base
            if not np.allclose(cal_g, cur_g, atol=0.1):
                logger.warning(
                    f"Calibration was performed with gravity={cal_g.tolist()} "
                    f"but current mounting has gravity={cur_g.tolist()}. "
                    f"Results may be inaccurate if mounting orientation changed. "
                    f"Consider re-calibrating with current mounting."
                )

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

    def _set_state_locked(self, state: ArmState) -> None:
        with self._lock:
            self._state = state

    def _check_enabled(self) -> None:
        if not self._enabled:
            raise OneroError("Arm is not enabled. Call arm.enable() first.")
        if self._safety.emergency_stopped:
            raise OneroError(
                "Arm is in emergency stop. Call arm.enable() to clear and re-enable.")

    def _check_no_freedrive(self) -> None:
        if self._freedrive_active:
            raise OneroError(
                "Cannot send commands while freedrive is active. "
                "Call freedrive_disable() first."
            )

    def _validate_joints(self, joints, param_name: str = "joints") -> list[float]:
        """Validate and convert joint array input. Returns list[float]."""
        if joints is None:
            raise OneroConfigError(
                f"'{param_name}' cannot be None. "
                f"Pass a list of {self.num_joints} joint angles in radians.")
        try:
            result = [float(v) for v in joints]
        except (TypeError, ValueError) as e:
            raise OneroConfigError(
                f"'{param_name}' must be a list of numbers. Got: {type(joints).__name__}. "
                f"Example: arm.movej([0.0, -0.5, 0.3, 0.5, -0.2, 0.3, 0.0])")
        if len(result) != self.num_joints:
            raise OneroConfigError(
                f"'{param_name}' must have {self.num_joints} elements "
                f"(one per joint), got {len(result)}")
        return result

    def _step_dt(self, dt: float) -> None:
        """Deadline-based sleep: accounts for time spent in command processing."""
        self._clock.tick(dt)

    def _begin_motion(self) -> None:
        """Acquire motion lock and reset state for a new planned motion."""
        if not self._motion_lock.acquire(blocking=False):
            raise OneroError(
                "Another motion is in progress. Wait for it to complete "
                "or call arm.stop() first.")
        self._clock.reset()
        self._force_mode_origin = None
        self._gain_policy.reset_cache()
        self._check_auto_payload()

    def _end_motion(self) -> None:
        """Release motion lock after motion completes."""
        if self._ik is not None:
            with self._lock:
                q = list(self._state.joints)
            self._ik.reset(np.array(q))
        try:
            self._motion_lock.release()
        except RuntimeError:
            pass

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

    def _parse_pose(self, method: str, position, rotation, frame=None):
        """Parse position/rotation arguments into (pos_3d, rot_matrix) in base frame."""
        pos_arr = np.asarray(position, dtype=float)
        if rotation is None:
            if pos_arr.shape == (6,):
                pos, rot = pos_arr[:3], geometry.rotvec_to_matrix(pos_arr[3:])
            elif pos_arr.shape == (3,):
                raise OneroConfigError(
                    f"{method}: 3D position requires rotation argument. "
                    "Pass rotation separately, or use a 6D pose [x,y,z,rx,ry,rz].")
            else:
                raise OneroConfigError(
                    f"{method}: expected position as [x,y,z] (3D) or "
                    f"[x,y,z,rx,ry,rz] (6D), got shape {pos_arr.shape}")
        else:
            pos, rot = pos_arr, geometry.ensure_rotation_matrix(rotation)
        return self._input_pose_transform(pos, rot, frame=frame)

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
            # Set gravity direction based on mounting orientation
            self._dynamics.set_gravity(self._gravity_in_base.tolist())
        return self._dynamics

    def _gravity_torques(self, joint_positions: list[float]) -> list[float]:
        """Compute gravity compensation torques via dynamics model (pinocchio)."""
        try:
            dynamics = self._get_dynamics()
            return dynamics.gravity_torques(joint_positions)
        except ImportError:
            if not getattr(self, '_gravity_warned', False):
                self._gravity_warned = True
                logger.warning(
                    "Dynamics model unavailable (pinocchio not installed). "
                    "Gravity compensation disabled — freedrive may not hold position. "
                    "Install with: pip install onero-sdk[dynamics]")
            return [0.0] * self.num_joints
        except Exception as e:
            logger.warning(f"Gravity torque computation failed: {e}. Using zero feedforward.")
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
        from .model_registry import _parse_model_variant
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        name = self._model_info.name
        base_model, variant = _parse_model_variant(name)
        candidates = [
            os.path.join(pkg_dir, "urdf", base_model, f"{base_model.lower()}_{variant}.urdf"),
            os.path.join(pkg_dir, "urdf", base_model, f"{base_model.lower()}_r.urdf"),
            os.path.join(pkg_dir, "urdf", base_model, f"{base_model.lower()}_l.urdf"),
            os.path.join(pkg_dir, "urdf", f"{base_model.lower()}.urdf"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None
