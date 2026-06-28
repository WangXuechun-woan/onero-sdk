"""
Generate GIF animations for README documentation.

Renders MuJoCo simulation offscreen with end-effector trail visualization,
producing GIF files showing each major SDK feature in action.

All demos use a workstation scene: arm side-mounted on a pillar with a
table in the workspace (matching real desktop installation).

All demos use the real SDK Arm API (same code path as real hardware).

Usage:
    python examples/generate_readme_gifs.py           # generate all
    python examples/generate_readme_gifs.py movej     # generate specific demo

Output:
    docs/assets/*.gif

Dependencies:
    pip install -e ".[full]"
    pip install imageio
"""
import sys
import os
import math
import numpy as np

os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, '.')

import mujoco
import imageio

from onero_sdk import Arm
from onero_sdk.backend import ArmState
from scipy.spatial.transform import Rotation

ASSETS_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'assets')
os.makedirs(ASSETS_DIR, exist_ok=True)

WIDTH = 640
HEIGHT = 480
FPS = 30

# Default camera for workstation scene (front-left 3/4 view)
DEFAULT_CAM_DISTANCE = 1.5
DEFAULT_CAM_AZIMUTH = -120
DEFAULT_CAM_ELEVATION = -20
DEFAULT_CAM_LOOKAT = [0.2, -0.1, 0.5]


# ─── Workstation Scene Backend ─────────────────────────────────────────────────

class WorkstationBackend:
    """
    MuJoCo backend with workstation scene: pillar + table + side-mounted arm.

    Builds the scene using MjSpec.attach() — same approach as DualMujocoBackend.
    The arm is mounted on a pillar in side-mount configuration, with workspace
    above a table (matching real desktop arm installations).
    """

    def __init__(self, model_name="A1R", control_dt=0.005, physics_dt=0.0005,
                 realtime=False, armature=0.01, on_step=None):
        self._control_dt = control_dt
        self._physics_dt = physics_dt
        self._substeps = max(1, int(round(control_dt / physics_dt)))
        self._realtime = realtime
        self._on_step = on_step
        self._step_count = 0
        self._nq = 7
        self._enabled = False

        self._model, self._data = self._build_scene(model_name, physics_dt, armature)

        from onero_sdk.model_registry import get_model
        model_info = get_model(model_name)
        self._torque_limits = np.array([j.t_max for j in model_info.joints]) * 2.0

        self.ee_body_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, 'Link7')
        self.link5_body_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, 'Link5')
        self.link4_body_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, 'Link4')
        self.link3_body_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, 'Link3')

    def _build_scene(self, model_name, physics_dt, armature):
        """Build workstation scene: ground + table + pillar + side-mounted arm."""
        from onero_sdk.model_registry import _parse_model_variant
        base_model, variant = _parse_model_variant(model_name)
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        sdk_dir = os.path.join(pkg_dir, '..', 'onero_sdk')
        urdf_dir = os.path.join(sdk_dir, 'urdf', base_model)
        r_urdf = os.path.join(urdf_dir, f"{base_model.lower()}_{variant}.urdf")

        spec = mujoco.MjSpec()
        spec.compiler.meshdir = urdf_dir

        # Ground plane
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[2.0, 2.0, 0.01],
            rgba=[0.92, 0.92, 0.90, 1.0],
        )

        # Lighting
        spec.worldbody.add_light(pos=[0.5, 0.5, 2.0], dir=[-0.3, -0.3, -1])
        spec.worldbody.add_light(pos=[-0.5, -0.5, 2.0], dir=[0.3, 0.3, -1])

        # Table (in front and slightly right of pillar)
        table = spec.worldbody.add_body(name='table', pos=[0.35, -0.15, 0.35])
        table.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.30, 0.35, 0.02],
            rgba=[0.6, 0.45, 0.3, 1.0],
        )
        for lx, ly in [(-0.25, -0.30), (-0.25, 0.30), (0.25, -0.30), (0.25, 0.30)]:
            table.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[0.02, 0.17],
                pos=[lx, ly, -0.17],
                rgba=[0.5, 0.35, 0.2, 1.0],
            )

        # Pillar/Stand
        pillar = spec.worldbody.add_body(name='pillar', pos=[0.0, 0.0, 0.0])
        # Main column
        pillar.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.04, 0.35],
            pos=[0, 0, 0.35],
            rgba=[0.4, 0.4, 0.45, 1.0],
        )
        # Base plate
        pillar.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.10, 0.015],
            pos=[0, 0, 0.015],
            rgba=[0.35, 0.35, 0.4, 1.0],
        )
        # Top bracket
        pillar.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.05, 0.05, 0.025],
            pos=[0, 0, 0.70],
            rgba=[0.35, 0.35, 0.4, 1.0],
        )

        # Mount site: euler=[90, 0, 180] makes arm reach forward & down
        # Arm Z → world -Y (side-mount-right), workspace above table
        mount_site = pillar.add_site(
            name='arm_mount', pos=[0, 0, 0.70], euler=[90, 0, 180])

        # Attach arm
        arm_spec = mujoco.MjSpec.from_file(r_urdf)
        spec.attach(arm_spec, prefix='', site=mount_site)

        # Compile
        model = spec.compile()
        model.opt.timestep = physics_dt
        model.dof_armature[:] = armature
        # Disable contacts (arm-only, no environment interaction)
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

        data = mujoco.MjData(model)
        return model, data

    @property
    def num_joints(self):
        return self._nq

    def enable(self):
        self._enabled = True
        mujoco.mj_resetData(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)
        return self._get_state()

    def disable(self):
        self._data.qfrc_applied[:self._nq] = 0.0
        self._enabled = False

    def command(self, positions, velocities, kps, kds, torques):
        pos_target = np.array(positions)
        vel_target = np.array(velocities)
        kp = np.array(kps)
        kd = np.array(kds)
        tau_ff = np.array(torques)

        for _ in range(self._substeps):
            q = self._data.qpos[:self._nq]
            v = self._data.qvel[:self._nq]
            tau = kp * (pos_target - q) + kd * (vel_target - v) + tau_ff
            np.clip(tau, -self._torque_limits, self._torque_limits, out=tau)
            self._data.qfrc_applied[:self._nq] = tau
            mujoco.mj_step(self._model, self._data)

        self._step_count += 1
        if self._on_step and self._step_count % 10 == 0:
            self._on_step()

        return self._get_state()

    def command_velocity(self, velocities):
        vel_target = np.array(velocities)
        kd_vel = 5.0

        for _ in range(self._substeps):
            v = self._data.qvel[:self._nq]
            tau = kd_vel * (vel_target - v)
            np.clip(tau, -self._torque_limits, self._torque_limits, out=tau)
            self._data.qfrc_applied[:self._nq] = tau
            mujoco.mj_step(self._model, self._data)

        self._step_count += 1
        if self._on_step and self._step_count % 10 == 0:
            self._on_step()

        return self._get_state()

    def command_torque(self, torques):
        tau = np.array(torques)
        np.clip(tau, -self._torque_limits, self._torque_limits, out=tau)

        for _ in range(self._substeps):
            self._data.qfrc_applied[:self._nq] = tau
            mujoco.mj_step(self._model, self._data)

        self._step_count += 1
        if self._on_step and self._step_count % 10 == 0:
            self._on_step()

        return self._get_state()

    def get_state(self):
        return self._get_state()

    def close(self):
        self._enabled = False

    def reset(self, positions=None):
        mujoco.mj_resetData(self._model, self._data)
        if positions is not None:
            self._data.qpos[:self._nq] = np.array(positions)
        mujoco.mj_forward(self._model, self._data)
        return self._get_state()

    def _get_state(self):
        return ArmState(
            joints=list(self._data.qpos[:self._nq]),
            velocities=list(self._data.qvel[:self._nq]),
            torques=list(self._data.qfrc_applied[:self._nq]),
            timestamp=self._data.time,
        )


# ─── Rendering helpers ──────────────────────────────────────────────────────

def render_scene(renderer, model, data, trail_points=None,
                 cam_distance=None, cam_azimuth=None, cam_elevation=None,
                 cam_lookat=None, trail_color=None, trail_colors=None,
                 arrows=None):
    """Render with camera settings, trail visualization, and optional arrows."""
    if cam_distance is None:
        cam_distance = DEFAULT_CAM_DISTANCE
    if cam_azimuth is None:
        cam_azimuth = DEFAULT_CAM_AZIMUTH
    if cam_elevation is None:
        cam_elevation = DEFAULT_CAM_ELEVATION
    if cam_lookat is None:
        cam_lookat = DEFAULT_CAM_LOOKAT
    if trail_color is None:
        trail_color = [1.0, 1.0, 0.0, 1.0]

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = cam_distance
    camera.azimuth = cam_azimuth
    camera.elevation = cam_elevation
    camera.lookat[:] = cam_lookat

    renderer.update_scene(data, camera=camera)

    scene = renderer.scene

    # Draw trail
    if trail_points and len(trail_points) > 0:
        max_trail = min(len(trail_points), scene.maxgeom - scene.ngeom - 50)
        if len(trail_points) > max_trail:
            indices = np.linspace(0, len(trail_points) - 1, max_trail, dtype=int)
            points = [trail_points[i] for i in indices]
            colors = [trail_colors[i] for i in indices] if trail_colors else None
        else:
            points = trail_points
            colors = trail_colors

        for k, pt in enumerate(points):
            if scene.ngeom >= scene.maxgeom - 50:
                break
            idx = scene.ngeom
            scene.ngeom += 1
            color = colors[k] if colors else trail_color
            mujoco.mjv_initGeom(
                scene.geoms[idx],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.008, 0.008, 0.008],
                pos=np.asarray(pt, dtype=np.float64),
                mat=np.eye(3).flatten(),
                rgba=np.asarray(color, dtype=np.float32),
            )
            scene.geoms[idx].emission = 1.0

    # Draw arrows (for force indicators)
    if arrows:
        for arrow in arrows:
            if scene.ngeom >= scene.maxgeom - 1:
                break
            idx = scene.ngeom
            scene.ngeom += 1
            from_pos = np.asarray(arrow['from'], dtype=np.float64)
            to_pos = np.asarray(arrow['to'], dtype=np.float64)
            diff = to_pos - from_pos
            length = np.linalg.norm(diff)
            if length < 1e-6:
                scene.ngeom -= 1
                continue
            mujoco.mjv_initGeom(
                scene.geoms[idx],
                type=mujoco.mjtGeom.mjGEOM_ARROW,
                size=[0.008, 0.008, length / 2],
                pos=(from_pos + to_pos) / 2,
                mat=np.eye(3).flatten(),
                rgba=np.asarray(arrow.get('color', [1, 0, 0, 1]), dtype=np.float32),
            )
            scene.geoms[idx].emission = 1.0

    return renderer.render().copy()


def get_ee_pos(data, body_id):
    """Get end-effector world position."""
    return data.xpos[body_id].copy()


def save_gif(frames, filename, fps=FPS):
    """Save frames as GIF with optimization."""
    path = os.path.join(ASSETS_DIR, filename)
    max_frames = fps * 8
    if len(frames) > max_frames:
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = [frames[i] for i in indices]
    imageio.mimsave(path, frames, fps=fps, loop=0)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  Saved: {path} ({len(frames)} frames, {size_mb:.1f} MB)")
    return path


class TrailRecorder:
    """Hook into backend.command to record trail and render frames."""

    def __init__(self, backend, renderer, trail_color=None, trail_colors_fn=None,
                 cam_distance=None, cam_azimuth=None, cam_elevation=None,
                 cam_lookat=None, arrows_fn=None):
        self.backend = backend
        self.renderer = renderer
        self.model = backend._model
        self.data = backend._data
        self.trail = []
        self.trail_colors = [] if trail_colors_fn else None
        self.frames = []
        self.trail_color = trail_color or [1.0, 1.0, 0.0, 1.0]
        self.trail_colors_fn = trail_colors_fn
        self.arrows_fn = arrows_fn
        self.step_count = 0
        self.render_every = max(1, int(1.0 / (FPS * backend._control_dt)))
        self._orig_command = backend.command
        self._orig_cmd_vel = backend.command_velocity
        self._orig_cmd_torque = backend.command_torque
        self._cam_distance = cam_distance or DEFAULT_CAM_DISTANCE
        self._cam_azimuth = cam_azimuth or DEFAULT_CAM_AZIMUTH
        self._cam_elevation = cam_elevation or DEFAULT_CAM_ELEVATION
        self._cam_lookat = cam_lookat or DEFAULT_CAM_LOOKAT
        self._ee_body_id = backend.ee_body_id

    def start(self):
        self.backend.command = self._hook_command
        self.backend.command_velocity = self._hook_velocity
        self.backend.command_torque = self._hook_torque

    def stop(self):
        self.backend.command = self._orig_command
        self.backend.command_velocity = self._orig_cmd_vel
        self.backend.command_torque = self._orig_cmd_torque

    def _record(self):
        self.trail.append(get_ee_pos(self.data, self._ee_body_id))
        if self.trail_colors_fn:
            self.trail_colors.append(self.trail_colors_fn())
        self.step_count += 1
        if self.step_count % self.render_every == 0:
            arrows = self.arrows_fn() if self.arrows_fn else None
            img = render_scene(
                self.renderer, self.model, self.data,
                trail_points=self.trail,
                trail_color=self.trail_color,
                trail_colors=self.trail_colors if self.trail_colors else None,
                cam_distance=self._cam_distance,
                cam_azimuth=self._cam_azimuth,
                cam_elevation=self._cam_elevation,
                cam_lookat=self._cam_lookat,
                arrows=arrows,
            )
            self.frames.append(img)

    def _hook_command(self, *args, **kwargs):
        result = self._orig_command(*args, **kwargs)
        self._record()
        return result

    def _hook_velocity(self, *args, **kwargs):
        result = self._orig_cmd_vel(*args, **kwargs)
        self._record()
        return result

    def _hook_torque(self, *args, **kwargs):
        result = self._orig_cmd_torque(*args, **kwargs)
        self._record()
        return result

    def hold_frame(self, seconds=1.0):
        """Add hold frames with the final trail visible."""
        for _ in range(int(FPS * seconds)):
            arrows = self.arrows_fn() if self.arrows_fn else None
            img = render_scene(
                self.renderer, self.model, self.data,
                trail_points=self.trail,
                trail_color=self.trail_color,
                trail_colors=self.trail_colors if self.trail_colors else None,
                cam_distance=self._cam_distance,
                cam_azimuth=self._cam_azimuth,
                cam_elevation=self._cam_elevation,
                cam_lookat=self._cam_lookat,
                arrows=arrows,
            )
            self.frames.append(img)


# ─── Helper: create standard workstation setup ─────────────────────────────

def create_workstation():
    """Create standard workstation backend + renderer + arm."""
    backend = WorkstationBackend(model_name="A1R", control_dt=0.005, realtime=False)
    renderer = mujoco.Renderer(backend._model, height=HEIGHT, width=WIDTH)
    arm = Arm("A1R", backend=backend)
    arm.enable()
    return backend, renderer, arm


# ─── Demo generators ────────────────────────────────────────────────────────

def demo_movej():
    """movej — joint-space motion with trapezoidal profile."""
    print("Generating movej.gif...")
    backend, renderer, arm = create_workstation()

    rec = TrailRecorder(backend, renderer, trail_color=[1.0, 1.0, 0.0, 1.0])
    rec.start()
    arm.movej([0.8, -1.0, 0.4, -1.5, 0.5, 0.4, -0.2], duration=3.0)
    rec.stop()
    rec.hold_frame(0.8)

    save_gif(rec.frames, 'movej.gif')
    arm.disable()
    renderer.close()


def demo_movel():
    """movel — Cartesian straight-line with IK. Clear long straight path."""
    print("Generating movel.gif...")
    backend, renderer, arm = create_workstation()

    arm.movej([0.0, -0.5, 0.0, -0.8, 0.0, 0.3, 0.0], duration=1.0)

    # Get starting EE pose in base frame (invariant to mounting)
    fk = arm._get_fk()
    start_pos, start_rot = fk.get_ee_pose(arm.state.joints)
    start_rotvec = Rotation.from_matrix(start_rot).as_rotvec()

    # Target: large straight-line motion (~200mm) in base frame
    target_pos = start_pos + np.array([0.12, -0.06, 0.14])
    pose_6d = np.concatenate([target_pos, start_rotvec])

    rec = TrailRecorder(backend, renderer, trail_color=[0.0, 1.0, 0.0, 1.0])
    rec.start()
    arm.movel(position=pose_6d, duration=3.0)
    rec.stop()
    rec.hold_frame(1.0)

    save_gif(rec.frames, 'movel.gif')
    arm.disable()
    renderer.close()


def demo_movec():
    """movec — circular arc. Clear arc shape with same orientation throughout."""
    print("Generating movec.gif...")
    backend, renderer, arm = create_workstation()

    arm.movej([0.0, -0.6, 0.0, -1.0, 0.0, 0.4, 0.0], duration=1.5)

    fk = arm._get_fk()
    start_pos, start_rot = fk.get_ee_pose(arm.state.joints)
    start_rotvec = Rotation.from_matrix(start_rot).as_rotvec()

    ik = arm._get_ik()
    ik.reset(np.array(arm.state.joints))

    # Arc in base frame (~140mm chord, ~74mm radius)
    via_pos = start_pos + np.array([0.06, 0.0, 0.10])
    target_pos = start_pos + np.array([0.14, 0.0, 0.01])

    via_pose = np.concatenate([via_pos, start_rotvec])
    target_pose = np.concatenate([target_pos, start_rotvec])

    rec = TrailRecorder(backend, renderer, trail_color=[1.0, 0.7, 0.0, 1.0])
    rec.start()
    arm.movec(via_pose, target_pose, duration=5.0)
    rec.stop()
    rec.hold_frame(1.0)

    save_gif(rec.frames, 'movec.gif')
    arm.disable()
    renderer.close()


def demo_servo_sine():
    """servoj — real-time streaming. Sine wave clearly visible in EE path."""
    print("Generating servo_sine.gif...")
    backend, renderer, arm = create_workstation()

    base_pose = [-0.35, -0.6, 0.0, -1.0, 0.0, 0.4, 0.0]
    arm.movej(base_pose, duration=1.0)

    rec = TrailRecorder(backend, renderer, trail_color=[0.0, 0.5, 1.0, 1.0],
                        cam_distance=1.6, cam_azimuth=-140, cam_elevation=-15,
                        cam_lookat=[0.15, -0.1, 0.5])
    rec.start()

    duration = 5.0
    steps = int(duration / backend._control_dt)

    for step in range(steps):
        t = step * backend._control_dt
        joints = list(base_pose)
        joints[0] = 0.7 * (t / duration) - 0.35
        joints[1] = base_pose[1] + 0.45 * math.sin(2 * math.pi * 0.8 * t)
        joints[3] = base_pose[3] - 0.2 * math.sin(2 * math.pi * 0.8 * t)
        arm.servoj(joints)

    rec.stop()
    rec.hold_frame(0.8)

    save_gif(rec.frames, 'servo_sine.gif')
    arm.disable()
    renderer.close()


def demo_blending():
    """Multi-segment blending: movel → movec → movel with smooth transitions."""
    print("Generating blending.gif...")
    backend, renderer, arm = create_workstation()

    arm.movej([0.0, -0.5, 0.0, -0.8, 0.0, 0.3, 0.0], duration=1.0)

    fk = arm._get_fk()
    _, start_rot = fk.get_ee_pose(arm.state.joints)
    start_rotvec = Rotation.from_matrix(start_rot).as_rotvec()
    ik = arm._get_ik()
    ik.reset(np.array(arm.state.joints))

    seg_colors = [
        [0.0, 1.0, 0.0, 1.0],   # green: straight line segment 1
        [1.0, 0.7, 0.0, 1.0],   # orange: arc segment
        [0.0, 0.8, 1.0, 1.0],   # cyan: straight line segment 2
        [1.0, 0.0, 1.0, 1.0],   # magenta: return
    ]
    current_seg = [0]

    def color_fn():
        return seg_colors[min(current_seg[0], len(seg_colors) - 1)]

    rec = TrailRecorder(backend, renderer, trail_colors_fn=color_fn)
    rec.start()

    # Segment 1: movel straight line (green) ~130mm in base frame
    current_seg[0] = 0
    pos, _ = fk.get_ee_pose(arm.state.joints)
    target1 = pos + np.array([0.08, 0.0, 0.10])
    arm.movel(np.concatenate([target1, start_rotvec]), duration=2.0)

    # Segment 2: movec arc (orange) ~110mm chord
    current_seg[0] = 1
    ik.reset(np.array(arm.state.joints))
    pos, _ = fk.get_ee_pose(arm.state.joints)
    via_pos = pos + np.array([0.05, 0.02, 0.06])
    target2 = pos + np.array([0.11, 0.0, 0.0])
    arm.movec(
        np.concatenate([via_pos, start_rotvec]),
        np.concatenate([target2, start_rotvec]),
        duration=3.5,
    )

    # Segment 3: movel straight line in different direction (cyan) ~100mm
    current_seg[0] = 2
    ik.reset(np.array(arm.state.joints))
    pos, _ = fk.get_ee_pose(arm.state.joints)
    target3 = pos + np.array([0.0, -0.06, 0.08])
    arm.movel(np.concatenate([target3, start_rotvec]), duration=2.0)

    # Segment 4: return to start (magenta)
    current_seg[0] = 3
    arm.movej([0.0, -0.3, 0.0, -0.5, 0.0, 0.1, 0.0], duration=2.0)

    rec.stop()
    rec.hold_frame(1.0)

    save_gif(rec.frames, 'blending.gif')
    arm.disable()
    renderer.close()


def demo_force_mode():
    """force_mode — hybrid force/position control."""
    print("Generating force_mode.gif...")
    backend, renderer, arm = create_workstation()

    arm.movej([0.0, -0.4, 0.0, -0.6, 0.0, 0.2, 0.0], duration=1.5)

    force_arrow = [None]

    def arrows_fn():
        if force_arrow[0] is not None:
            return [force_arrow[0]]
        return None

    rec = TrailRecorder(backend, renderer, trail_color=[1.0, 0.3, 0.0, 1.0],
                        arrows_fn=arrows_fn)
    rec.start()

    dt = backend._control_dt
    ee_bid = backend.ee_body_id

    # Phase 1: Push down with large force (2.5s)
    steps_down = int(2.5 / dt)
    for step in range(steps_down):
        t = step * dt
        fz = -30.0 * min(1.0, t / 0.5)
        arm.force_mode(
            selection=[0, 0, 1, 0, 0, 0],
            wrench=[0, 0, fz, 0, 0, 0],
            task_frame=[0, 0, 0, 0, 0, 0],
            limits=[0, 0, 0.25, 0, 0, 0],
        )
        ee_pos = get_ee_pos(backend._data, ee_bid)
        arrow_len = abs(fz) * 0.004
        force_arrow[0] = {
            'from': ee_pos,
            'to': ee_pos + np.array([0, 0, -arrow_len]),
            'color': [1.0, 0.2, 0.2, 1.0],
        }

    # Phase 2: Slide laterally while maintaining force (3.5s)
    steps_slide = int(3.5 / dt)
    for step in range(steps_slide):
        fx = 25.0
        fz = -30.0
        arm.force_mode(
            selection=[1, 0, 1, 0, 0, 0],
            wrench=[fx, 0, fz, 0, 0, 0],
            task_frame=[0, 0, 0, 0, 0, 0],
            limits=[0.20, 0, 0.25, 0, 0, 0],
        )
        ee_pos = get_ee_pos(backend._data, ee_bid)
        force_arrow[0] = {
            'from': ee_pos,
            'to': ee_pos + np.array([fx * 0.005, 0, fz * 0.005]),
            'color': [1.0, 0.2, 0.2, 1.0],
        }

    arm.force_mode_disable()
    force_arrow[0] = None
    rec.stop()
    rec.hold_frame(0.5)

    save_gif(rec.frames, 'force_mode.gif')
    arm.disable()
    renderer.close()


def demo_freedrive():
    """freedrive — teach mode + replay + 2x replay."""
    print("Generating freedrive.gif...")
    backend, renderer, arm = create_workstation()

    start_pose = [0.2, -0.4, 0.1, -0.6, 0.0, 0.2, 0.0]
    arm.movej(start_pose, duration=1.0)

    seg_colors = [
        [1.0, 0.3, 1.0, 1.0],  # magenta: freedrive/teach
        [0.0, 1.0, 0.3, 1.0],  # green: replay 1x
        [0.0, 0.8, 1.0, 1.0],  # cyan: replay 2x
    ]
    current_seg = [0]

    def color_fn():
        return seg_colors[min(current_seg[0], len(seg_colors) - 1)]

    rec = TrailRecorder(backend, renderer, trail_colors_fn=color_fn)
    rec.start()

    # Phase 1: Freedrive (teach) — record while "human" drags the arm
    current_seg[0] = 0
    n = arm.num_joints
    damping = 0.8
    teach_duration = 3.5
    steps = int(teach_duration / backend._control_dt)

    arm.start_recording()

    link5_bid = backend.link5_body_id
    ee_bid = backend.ee_body_id

    for step in range(steps):
        t = step * backend._control_dt
        backend._data.xfrc_applied[link5_bid] = [
            1.2 * math.sin(2 * math.pi * 0.2 * t),
            0.8 * math.sin(2 * math.pi * 0.15 * t),
            0.6 * math.cos(2 * math.pi * 0.12 * t), 0, 0, 0]
        backend._data.xfrc_applied[ee_bid] = [
            0.6 * math.sin(2 * math.pi * 0.25 * t + 2.0),
            0, 0.5 * math.cos(2 * math.pi * 0.18 * t), 0, 0, 0]

        q = list(backend._data.qpos[:n])
        g = arm.gravity_torque()
        arm.servoj(q, kp=0.0, kd=damping, tau_ff=g)

    backend._data.xfrc_applied[:] = 0
    recording = arm.stop_recording()

    # Fix timestamps for non-realtime sim
    from onero_sdk.trajectory import Recording, RecordedFrame
    fixed_frames = []
    for i, f in enumerate(recording.frames):
        fixed_frames.append(RecordedFrame(
            timestamp=i * backend._control_dt,
            joints=f.joints, velocities=f.velocities, torques=f.torques,
        ))
    recording = Recording(fixed_frames)

    # Phase 2: Replay at 1x speed (green trail)
    current_seg[0] = 1
    rec.stop()
    arm.movej(start_pose, duration=1.0)
    rec.start()
    arm.replay(recording, speed_factor=1.0, kp=120.0, kd=2.0)

    # Phase 3: Replay at 2x speed (cyan trail)
    current_seg[0] = 2
    rec.stop()
    arm.movej(start_pose, duration=1.0)
    rec.start()
    arm.replay(recording, speed_factor=2.0, kp=120.0, kd=2.0)

    rec.stop()
    rec.hold_frame(1.0)

    save_gif(rec.frames, 'freedrive.gif')
    arm.disable()
    renderer.close()


def demo_speed_control():
    """speedj + speedl — joint velocity vs Cartesian velocity."""
    print("Generating speedj.gif...")
    backend, renderer, arm = create_workstation()

    current_color = [None]
    color_cyan = [0.0, 1.0, 0.8, 1.0]
    color_yellow = [1.0, 1.0, 0.0, 1.0]
    current_color[0] = color_cyan

    rec = TrailRecorder(backend, renderer,
                        trail_colors_fn=lambda: list(current_color[0]))
    rec.start()

    # Phase 1: speedj — joint velocity produces arc at EE (cyan)
    arm.speedj([0.0, 0.5, 0.0, -0.3, 0.0, 0.2, 0.0], acceleration=1.0, duration=3.0)

    # Phase 2: speedl — Cartesian velocity produces straight line (yellow)
    current_color[0] = color_yellow
    arm.speedl([0.0, 0.0, -0.08, 0.0, 0.0, 0.0], acceleration=0.5, duration=2.5)

    rec.stop()
    rec.hold_frame(0.5)

    save_gif(rec.frames, 'speedj.gif')
    arm.disable()
    renderer.close()


def demo_impedance():
    """Impedance control — compliant response to external perturbation."""
    print("Generating impedance.gif...")
    backend, renderer, arm = create_workstation()

    init_pose = [0.3, -0.5, 0.2, -0.8, 0.2, 0.2, 0.0]
    arm.movej(init_pose, duration=1.5)

    rec = TrailRecorder(backend, renderer, trail_color=[0.7, 0.3, 1.0, 1.0])
    rec.start()

    n = arm.num_joints
    duration = 4.0
    steps = int(duration / backend._control_dt)

    link5_bid = backend.link5_body_id
    link3_bid = backend.link3_body_id

    for step in range(steps):
        t = step * backend._control_dt
        backend._data.xfrc_applied[link5_bid] = [
            5.0 * math.sin(2 * math.pi * 0.3 * t), 0,
            5.0 * math.cos(2 * math.pi * 0.4 * t), 0, 0, 0]
        backend._data.xfrc_applied[link3_bid] = [
            0, 3.5 * math.sin(2 * math.pi * 0.5 * t), 0, 0, 0, 0]

        g = arm.gravity_torque()
        arm.servoj(init_pose, kp=8.0, kd=1.0, tau_ff=g)

    backend._data.xfrc_applied[:] = 0
    rec.stop()
    rec.hold_frame(0.5)

    save_gif(rec.frames, 'impedance.gif')
    arm.disable()
    renderer.close()


def demo_nullspace():
    """Null-space motion — 7-DOF redundancy demonstration."""
    print("Generating nullspace.gif...")
    backend, renderer, arm = create_workstation()

    arm.movej([0.0, -1.0, 0.0, -1.5, 0.0, 0.5, 0.0], duration=1.5)

    dynamics = arm._get_dynamics()
    fk = arm._get_fk()
    ik = arm._get_ik()
    dt = backend._control_dt
    n = arm.num_joints

    q_min = np.array([-1.04, -3.14, -1.04, -3.14, -1.04, -3.14, -1.04])
    q_max = np.array([3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14])
    margin = 0.1

    ee_target_pos, ee_target_rot = fk.get_ee_pose(list(arm.state.joints))

    q_excite1 = np.array([0.4, -0.3, 1.2, 0.3, -1.0, 0.15, 0.6])
    q_excite2 = np.array([-0.3, 0.5, -0.6, 1.0, 0.4, -0.7, 0.3])
    q_current = np.array(arm.state.joints)
    ee_fixed_pos = get_ee_pos(backend._data, backend.ee_body_id)

    elbow_trail = []
    frames = []
    render_every = max(1, int(1.0 / (FPS * dt)))
    step_count = 0

    link4_bid = backend.link4_body_id

    duration = 15.0
    steps = int(duration / dt)

    for step in range(steps):
        t = step * dt
        amp1 = 4.0 * math.sin(2 * math.pi * 0.12 * t)
        amp2 = 2.5 * math.sin(2 * math.pi * 0.2 * t + 1.5)
        q_excite = q_excite1 * amp1 + q_excite2 * amp2

        J = dynamics.jacobian(list(q_current))
        J_pinv = np.linalg.pinv(J)
        N = np.eye(n) - J_pinv @ J
        dq_null = N @ q_excite

        q_candidate = q_current + dq_null * dt
        q_candidate = np.clip(q_candidate, q_min + margin, q_max - margin)

        ik.reset(q_candidate)
        q_current = np.array(ik.solve(ee_target_pos, ee_target_rot, dt=dt))

        arm.servoj(list(q_current), kp=100.0, kd=2.0)

        if step % 3 == 0:
            elbow_trail.append(backend._data.xpos[link4_bid].copy())

        step_count += 1
        if step_count % render_every == 0:
            trail_pts = elbow_trail + [ee_fixed_pos]
            trail_cols = (
                [[1.0, 0.0, 1.0, 1.0]] * len(elbow_trail) +
                [[0.0, 1.0, 0.0, 1.0]]
            )
            img = render_scene(
                renderer, backend._model, backend._data,
                trail_points=trail_pts,
                trail_colors=trail_cols,
                cam_distance=1.4,
                cam_azimuth=-130,
                cam_elevation=-20,
                cam_lookat=DEFAULT_CAM_LOOKAT,
            )
            frames.append(img)

    # Hold final frame
    for _ in range(int(FPS * 2.0)):
        trail_pts = elbow_trail + [ee_fixed_pos]
        trail_cols = (
            [[1.0, 0.0, 1.0, 1.0]] * len(elbow_trail) +
            [[0.0, 1.0, 0.0, 1.0]]
        )
        img = render_scene(
            renderer, backend._model, backend._data,
            trail_points=trail_pts,
            trail_colors=trail_cols,
            cam_distance=1.4,
            cam_azimuth=-130,
            cam_elevation=-20,
            cam_lookat=DEFAULT_CAM_LOOKAT,
        )
        frames.append(img)

    save_gif(frames, 'nullspace.gif')
    arm.disable()
    renderer.close()


def demo_trajectory_profiles():
    """Trajectory profiles — trapezoidal velocity profile with large motion."""
    print("Generating trajectory_profiles.gif...")
    backend, renderer, arm = create_workstation()

    target = [0.8, -0.7, 0.4, -1.2, 0.3, 0.2, -0.15]

    rec = TrailRecorder(backend, renderer, trail_color=[0.0, 1.0, 0.0, 1.0])
    rec.start()
    arm.movej(target, duration=2.5)
    rec.stop()
    rec.hold_frame(0.5)

    save_gif(rec.frames, 'trajectory_profiles.gif')
    arm.disable()
    renderer.close()


def demo_dual_arm():
    """Dual arm — mirror motion of two arms."""
    print("Generating dual_arm.gif...")
    from onero_sdk.dual_mujoco_backend import DualMujocoBackend

    backend = DualMujocoBackend(model_name="A1R", realtime=False, control_dt=0.005)
    renderer = mujoco.Renderer(backend._model, height=HEIGHT, width=WIDTH)

    frames = []
    render_every = max(1, int(1.0 / (FPS * backend._control_dt)))
    step_counter = [0]

    orig_step = backend.step

    def hook_step(*args, **kwargs):
        result = orig_step(*args, **kwargs)
        step_counter[0] += 1
        if step_counter[0] % render_every == 0:
            img = render_scene(renderer, backend._model, backend._data,
                               cam_distance=2.0, cam_azimuth=-150,
                               cam_elevation=-20, cam_lookat=[0, 0, 0.9])
            frames.append(img)
        return result

    backend.step = hook_step

    poses = [
        ([1.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0]),
        ([0.5, -0.5, 0.3, -1.0, 0.2, 0.2, 0.0], [-0.5, 0.5, -0.3, 1.0, -0.2, -0.2, 0.0]),
        ([1.2, -0.3, 0.0, -0.8, 0.4, 0.0, 0.0], [-1.2, 0.3, 0.0, 0.8, -0.4, 0.0, 0.0]),
        ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ]

    for r_target, l_target in poses:
        backend.movej(r_target, l_target, duration=1.5)

    backend.step = orig_step
    save_gif(frames, 'dual_arm.gif')
    renderer.close()


# ─── Main ───────────────────────────────────────────────────────────────────

ALL_DEMOS = {
    'movej': demo_movej,
    'movel': demo_movel,
    'servo_sine': demo_servo_sine,
    'movec': demo_movec,
    'freedrive': demo_freedrive,
    'speedj': demo_speed_control,
    'blending': demo_blending,
    'trajectory_profiles': demo_trajectory_profiles,
    'force_mode': demo_force_mode,
    'dual_arm': demo_dual_arm,
    'impedance': demo_impedance,
    'nullspace': demo_nullspace,
}


def main():
    print("=" * 60)
    print("  Generating README GIF animations (workstation scene)")
    print("=" * 60)
    print()

    if len(sys.argv) > 1:
        names = sys.argv[1:]
    else:
        names = list(ALL_DEMOS.keys())

    for name in names:
        if name in ALL_DEMOS:
            ALL_DEMOS[name]()
        else:
            print(f"  Unknown demo: {name}")
            print(f"  Available: {', '.join(ALL_DEMOS.keys())}")

    print()
    print("=" * 60)
    print(f"  All GIFs saved to: {os.path.abspath(ASSETS_DIR)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
