"""
Generate GIF animations for README documentation.

Renders MuJoCo simulation offscreen with end-effector trail visualization,
producing GIF files showing each major SDK feature in action.

All demos use a dual-arm torso scene (human-like upper body + table),
with only the right arm executing motions while the left arm holds at zero.

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
from dataclasses import dataclass

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


# ─── Scene Geometry (adaptive, read from MuJoCo model) ───────────────────────

@dataclass
class SceneGeometry:
    """Geometry extracted from the MuJoCo scene for adaptive pose computation."""
    table_z: float            # table surface height (world frame)
    table_x_range: tuple      # (x_min, x_max) of table top
    table_y_range: tuple      # (y_min, y_max) of table top
    shoulder_pos: np.ndarray  # right shoulder world position


def get_scene_geometry(model, data):
    """Extract table and arm mounting geometry from MuJoCo model."""
    # Table: find body, read position + largest box geom half-extents
    table_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'table')
    table_pos = data.xpos[table_bid].copy()
    # Find the table-top geom (BOX type with largest Z half-extent among table geoms)
    best_hz = 0
    half_extents = np.zeros(3)
    for i in range(model.ngeom):
        if model.geom_bodyid[i] == table_bid and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX:
            hz = model.geom_size[i, 2]
            if hz >= best_hz:
                best_hz = hz
                half_extents = model.geom_size[i].copy()

    table_surface_z = table_pos[2] + half_extents[2]
    table_x_range = (table_pos[0] - half_extents[0], table_pos[0] + half_extents[0])
    table_y_range = (table_pos[1] - half_extents[1], table_pos[1] + half_extents[1])

    # Right shoulder: find R_base_link body position in world
    r_base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'R_base_link')
    shoulder_pos = data.xpos[r_base_bid].copy()

    return SceneGeometry(
        table_z=table_surface_z,
        table_x_range=table_x_range,
        table_y_range=table_y_range,
        shoulder_pos=shoulder_pos,
    )


def compute_camera_params(scene_geo):
    """Derive default camera parameters from scene geometry."""
    lookat_z = float(scene_geo.shoulder_pos[2])
    return {
        'distance': 1.8,
        'azimuth': -140,
        'elevation': -20,
        'lookat': [0.2, -0.1, lookat_z],
    }


# ─── Dual-Scene Right-Arm Adapter ─────────────────────────────────────────────

class DualSceneAdapter:
    """
    Wraps DualMujocoBackend to expose a single-arm ArmBackend interface.

    Only the right arm is controlled; the left arm holds at zero with PD.
    Exposes _model, _data, ee_body_id, etc. for TrailRecorder compatibility.
    """

    def __init__(self, model_name="A1R", control_dt=0.005):
        from onero_sdk.dual_mujoco_backend import DualMujocoBackend

        self._dual = DualMujocoBackend(
            model_name=model_name, realtime=False, control_dt=control_dt)
        self._control_dt = control_dt
        self._nq = 7
        self._enabled = False
        self._left_target = np.zeros(7)

        self._model = self._dual._model
        self._data = self._dual._data

        self.ee_body_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, 'R_Link7')
        self.link5_body_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, 'R_Link5')
        self.link4_body_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, 'R_Link4')
        self.link3_body_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, 'R_Link3')

        from onero_sdk.model_registry import get_model
        model_info = get_model(model_name)
        self._torque_limits = np.array([j.t_max for j in model_info.joints]) * 20.0

    def gravity_torques(self, joint_positions):
        """Compute gravity using MuJoCo's dynamics (accounts for mounting orientation)."""
        saved_qpos = self._data.qpos[:7].copy()
        saved_qvel = self._data.qvel[:7].copy()
        self._data.qpos[:7] = joint_positions
        self._data.qvel[:7] = 0
        mujoco.mj_forward(self._model, self._data)
        g = self._data.qfrc_bias[:7].copy()
        self._data.qpos[:7] = saved_qpos
        self._data.qvel[:7] = saved_qvel
        mujoco.mj_forward(self._model, self._data)
        return g.tolist()

    @property
    def num_joints(self):
        return self._nq

    def enable(self):
        self._enabled = True
        self._dual.reset()
        return self._get_state()

    def disable(self):
        self._enabled = False

    def command(self, positions, velocities, kps, kds, torques):
        pos_target = np.array(positions)
        vel_target = np.array(velocities)
        kp = np.array(kps)
        kd = np.array(kds)
        tau_ff = np.array(torques)

        substeps = self._dual._substeps
        torque_lim = np.concatenate([self._torque_limits, self._torque_limits])

        for _ in range(substeps):
            q = self._data.qpos[:14]
            v = self._data.qvel[:14]

            # Right arm: user-specified gains
            r_tau = kp * (pos_target - q[:7]) + kd * (vel_target - v[:7]) + tau_ff
            # Left arm: hold at zero
            l_tau = 150.0 * (self._left_target - q[7:14]) + 5.0 * (0.0 - v[7:14])

            tau = np.concatenate([r_tau, l_tau])
            np.clip(tau, -torque_lim, torque_lim, out=tau)
            self._data.qfrc_applied[:14] = tau
            mujoco.mj_step(self._model, self._data)

        return self._get_state()

    def command_velocity(self, velocities):
        vel_target = np.array(velocities)
        substeps = self._dual._substeps
        torque_lim = np.concatenate([self._torque_limits, self._torque_limits])

        for _ in range(substeps):
            q = self._data.qpos[:14]
            v = self._data.qvel[:14]

            r_tau = 5.0 * (vel_target - v[:7])
            l_tau = 150.0 * (self._left_target - q[7:14]) + 5.0 * (0.0 - v[7:14])

            tau = np.concatenate([r_tau, l_tau])
            np.clip(tau, -torque_lim, torque_lim, out=tau)
            self._data.qfrc_applied[:14] = tau
            mujoco.mj_step(self._model, self._data)

        return self._get_state()

    def command_torque(self, torques):
        tau_r = np.array(torques)
        np.clip(tau_r, -self._torque_limits, self._torque_limits, out=tau_r)
        substeps = self._dual._substeps
        torque_lim = np.concatenate([self._torque_limits, self._torque_limits])

        for _ in range(substeps):
            q = self._data.qpos[:14]
            v = self._data.qvel[:14]

            l_tau = 150.0 * (self._left_target - q[7:14]) + 5.0 * (0.0 - v[7:14])
            tau = np.concatenate([tau_r, l_tau])
            np.clip(tau, -torque_lim, torque_lim, out=tau)
            self._data.qfrc_applied[:14] = tau
            mujoco.mj_step(self._model, self._data)

        return self._get_state()

    def get_state(self):
        return self._get_state()

    def close(self):
        self._enabled = False

    def reset(self, positions=None):
        self._dual.reset(right_pos=positions)
        return self._get_state()

    def _get_state(self):
        return ArmState(
            joints=list(self._data.qpos[:7]),
            velocities=list(self._data.qvel[:7]),
            torques=list(self._data.qfrc_applied[:7]),
            timestamp=self._data.time,
        )


# ─── Adaptive pose helpers ─────────────────────────────────────────────────

def teleport_arm(arm, backend, positions):
    """Reset arm to target position instantly, bypassing physics simulation."""
    backend.reset(positions=positions)
    state = backend.get_state()
    arm._state = state
    arm._safety.reset(state.joints)


def ee_pos_at(backend, positions):
    """Get EE world position for given joint config (temporary FK via MuJoCo)."""
    saved = backend._data.qpos[:7].copy()
    backend._data.qpos[:7] = positions
    mujoco.mj_forward(backend._model, backend._data)
    ee = backend._data.xpos[backend.ee_body_id].copy()
    backend._data.qpos[:7] = saved
    mujoco.mj_forward(backend._model, backend._data)
    return ee


def verify_above_table(backend, positions, scene_geo, margin=0.04):
    """Check that EE at given joint positions is above table surface."""
    ee = ee_pos_at(backend, positions)
    return ee[2] > scene_geo.table_z + margin


def find_home_pose(backend, scene_geo, margin=0.06):
    """Find a joint config placing EE above table with clearance.

    Searches J0/J3 (the two joints most affecting EE height in this mounting)
    for the configuration closest to above-table-center.
    """
    best_pose = None
    best_score = float('inf')
    table_cx = (scene_geo.table_x_range[0] + scene_geo.table_x_range[1]) / 2
    table_cy = (scene_geo.table_y_range[0] + scene_geo.table_y_range[1]) / 2
    target_z = scene_geo.table_z + margin

    for j0 in np.arange(0.3, 1.3, 0.1):
        for j3 in np.arange(0.5, 1.5, 0.1):
            pose = [j0, 0.0, 0.0, j3, 0.0, 0.5, 0.0]
            ee = ee_pos_at(backend, pose)
            if ee[2] < target_z:
                continue
            dist = math.hypot(ee[0] - table_cx, ee[1] - table_cy)
            z_excess = ee[2] - target_z
            score = dist + 0.5 * z_excess
            if score < best_score:
                best_score = score
                best_pose = pose

    if best_pose is None:
        best_pose = [0.6, 0.0, 0.0, 1.0, 0.0, 0.5, 0.0]
    return best_pose


def compute_demo_poses(backend, scene_geo, home):
    """Compute all demo start/target poses relative to home, verified above table."""
    margin = 0.04

    def offset_pose(offsets):
        return [home[i] + offsets[i] for i in range(7)]

    def verified(pose, label=""):
        if verify_above_table(backend, pose, scene_geo, margin):
            return pose
        print(f"  WARNING: {label} EE below table, scaling down offset")
        # Fallback: lerp toward home until safe
        for alpha in [0.7, 0.5, 0.3, 0.0]:
            mixed = [home[i] * (1 - alpha) + pose[i] * alpha for i in range(7)]
            if verify_above_table(backend, mixed, scene_geo, margin):
                return mixed
        return home

    poses = {}
    poses['home'] = home
    poses['movej_start'] = verified(offset_pose([-0.4, 0, 0, 0.5, 0, 0, 0]), "movej_start")
    poses['movej_target'] = verified(offset_pose([1.0, 0, 0, -1.5, 0, 0, 0]), "movej_target")
    poses['movec_start'] = verified(offset_pose([0.2, 0, 0.2, -0.2, 0, -0.1, 0]), "movec_start")
    poses['servo_sine_base'] = verified(offset_pose([0.0, 0.1, 0, 0, 0, 0, 0]), "servo_sine")
    poses['speedj_start'] = verified(offset_pose([0.0, 0.1, 0, 0, 0, 0, 0]), "speedj")
    poses['impedance_start'] = verified(offset_pose([0.2, 0.1, -0.2, 0, 0.2, 0.1, -0.1]), "impedance")
    poses['nullspace_start'] = verified(offset_pose([0, 0, 0, 0.1, 0, 0, 0]), "nullspace")
    poses['traj_target'] = verified(offset_pose([0.6, 0, 0, -0.8, 0, 0, 0]), "traj_target")
    return poses


# ─── Module-level scene state (initialized once by create_workstation) ──────

_scene_geo = None
_cam_params = None
_demo_poses = None


# ─── Rendering helpers ──────────────────────────────────────────────────────

def render_scene(renderer, model, data, trail_points=None,
                 cam_distance=None, cam_azimuth=None, cam_elevation=None,
                 cam_lookat=None, trail_color=None, trail_colors=None,
                 arrows=None):
    """Render with camera settings, trail visualization, and optional arrows."""
    if cam_distance is None:
        cam_distance = _cam_params['distance']
    if cam_azimuth is None:
        cam_azimuth = _cam_params['azimuth']
    if cam_elevation is None:
        cam_elevation = _cam_params['elevation']
    if cam_lookat is None:
        cam_lookat = _cam_params['lookat']
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
        self._cam_distance = cam_distance or _cam_params['distance']
        self._cam_azimuth = cam_azimuth or _cam_params['azimuth']
        self._cam_elevation = cam_elevation or _cam_params['elevation']
        self._cam_lookat = cam_lookat or _cam_params['lookat']
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
    """Create dual-arm scene with right-arm adapter + renderer + arm.

    Initializes adaptive scene geometry and computes safe demo poses on first call.
    """
    global _scene_geo, _cam_params, _demo_poses

    backend = DualSceneAdapter(model_name="A1R", control_dt=0.005)
    renderer = mujoco.Renderer(backend._model, height=HEIGHT, width=WIDTH)
    arm = Arm("A1R", backend=backend)
    arm._gravity_torques = backend.gravity_torques
    arm._executor._gravity_torques_fn = backend.gravity_torques
    arm.enable()

    if _scene_geo is None:
        mujoco.mj_forward(backend._model, backend._data)
        _scene_geo = get_scene_geometry(backend._model, backend._data)
        _cam_params = compute_camera_params(_scene_geo)
        home = find_home_pose(backend, _scene_geo)
        _demo_poses = compute_demo_poses(backend, _scene_geo, home)

    return backend, renderer, arm


# ─── Demo generators ────────────────────────────────────────────────────────

def demo_movej():
    """movej — joint-space motion with trapezoidal profile."""
    print("Generating movej.gif...")
    backend, renderer, arm = create_workstation()

    teleport_arm(arm, backend, _demo_poses['movej_start'])

    rec = TrailRecorder(backend, renderer, trail_color=[1.0, 1.0, 0.0, 1.0])
    rec.start()
    arm.movej(_demo_poses['movej_target'], duration=3.0)
    rec.stop()
    rec.hold_frame(0.8)

    save_gif(rec.frames, 'movej.gif')
    arm.disable()
    renderer.close()


def demo_movel():
    """movel — Cartesian straight-line with IK. Clear long straight path."""
    print("Generating movel.gif...")
    backend, renderer, arm = create_workstation()

    teleport_arm(arm, backend, _demo_poses['home'])

    # Get starting EE pose in base frame (invariant to mounting)
    fk = arm._get_fk()
    start_pos, start_rot = fk.get_ee_pose(arm.state.joints)
    start_rotvec = Rotation.from_matrix(start_rot).as_rotvec()

    # Target: large straight-line motion (~200mm) in base frame
    target_pos = start_pos + np.array([0.10, -0.05, 0.08])
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

    teleport_arm(arm, backend, _demo_poses['movec_start'])

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

    base_pose = _demo_poses['servo_sine_base']
    teleport_arm(arm, backend, base_pose)

    rec = TrailRecorder(backend, renderer, trail_color=[0.0, 0.5, 1.0, 1.0],
                        cam_distance=1.9, cam_azimuth=-140, cam_elevation=-15,
                        cam_lookat=[0.15, -0.1, 0.85])
    rec.start()

    duration = 5.0
    steps = int(duration / backend._control_dt)

    for step in range(steps):
        t = step * backend._control_dt
        joints = list(base_pose)
        joints[0] = base_pose[0] + 0.8 * (t / duration)
        joints[3] = base_pose[3] + 0.25 * math.sin(2 * math.pi * 0.8 * t)
        joints[5] = base_pose[5] + 0.15 * math.sin(2 * math.pi * 1.2 * t + 1.0)
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

    teleport_arm(arm, backend, _demo_poses['home'])

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

    # Segment 1: movel straight line (green) ~100mm in base frame
    current_seg[0] = 0
    pos, _ = fk.get_ee_pose(arm.state.joints)
    target1 = pos + np.array([0.05, 0.0, 0.07])
    arm.movel(np.concatenate([target1, start_rotvec]), duration=2.0)

    # Segment 2: movec arc (orange) ~80mm chord
    current_seg[0] = 1
    ik.reset(np.array(arm.state.joints))
    pos, _ = fk.get_ee_pose(arm.state.joints)
    via_pos = pos + np.array([0.03, 0.02, 0.04])
    target2 = pos + np.array([0.07, 0.0, 0.0])
    arm.movec(
        np.concatenate([via_pos, start_rotvec]),
        np.concatenate([target2, start_rotvec]),
        duration=3.5,
    )

    # Segment 3: movel straight line in different direction (cyan) ~50mm
    current_seg[0] = 2
    ik.reset(np.array(arm.state.joints))
    pos, _ = fk.get_ee_pose(arm.state.joints)
    target3 = pos + np.array([0.0, -0.03, 0.03])
    arm.movel(np.concatenate([target3, start_rotvec]), duration=2.0)

    # Segment 4: return to start (magenta)
    current_seg[0] = 3
    arm.movej(_demo_poses['home'], duration=2.0)

    rec.stop()
    rec.hold_frame(1.0)

    save_gif(rec.frames, 'blending.gif')
    arm.disable()
    renderer.close()


def demo_force_mode():
    """force_mode — hybrid force/position control."""
    print("Generating force_mode.gif...")
    backend, renderer, arm = create_workstation()

    teleport_arm(arm, backend, _demo_poses['home'])

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

    # Phase 1: Push forward with force (2.5s)
    steps_down = int(2.5 / dt)
    for step in range(steps_down):
        t = step * dt
        fx = 25.0 * min(1.0, t / 0.5)
        arm.force_mode(
            selection=[1, 0, 0, 0, 0, 0],
            wrench=[fx, 0, 0, 0, 0, 0],
            task_frame=[0, 0, 0, 0, 0, 0],
            limits=[0.15, 0, 0, 0, 0, 0],
        )
        ee_pos = get_ee_pos(backend._data, ee_bid)
        arrow_len = abs(fx) * 0.004
        force_arrow[0] = {
            'from': ee_pos,
            'to': ee_pos + np.array([0, 0, -arrow_len]),
            'color': [1.0, 0.2, 0.2, 1.0],
        }

    # Phase 2: Slide laterally while maintaining forward force (3.5s)
    steps_slide = int(3.5 / dt)
    for step in range(steps_slide):
        fx = 20.0
        fy = 15.0
        arm.force_mode(
            selection=[1, 1, 0, 0, 0, 0],
            wrench=[fx, fy, 0, 0, 0, 0],
            task_frame=[0, 0, 0, 0, 0, 0],
            limits=[0.15, 0.12, 0, 0, 0, 0],
        )
        ee_pos = get_ee_pos(backend._data, ee_bid)
        force_arrow[0] = {
            'from': ee_pos,
            'to': ee_pos + np.array([fx * 0.004, fy * 0.004, 0]),
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

    start_pose = _demo_poses['home']
    teleport_arm(arm, backend, start_pose)

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
            0.8 * math.sin(2 * math.pi * 0.2 * t),
            0.5 * math.sin(2 * math.pi * 0.15 * t),
            0.4 * math.cos(2 * math.pi * 0.12 * t), 0, 0, 0]
        backend._data.xfrc_applied[ee_bid] = [
            0.4 * math.sin(2 * math.pi * 0.25 * t + 2.0),
            0, 0.3 * math.cos(2 * math.pi * 0.18 * t), 0, 0, 0]

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
    teleport_arm(arm, backend, start_pose)
    rec.start()
    arm.replay(recording, speed_factor=1.0, kp=120.0, kd=2.0)

    # Phase 3: Replay at 2x speed (cyan trail)
    current_seg[0] = 2
    rec.stop()
    teleport_arm(arm, backend, start_pose)
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

    teleport_arm(arm, backend, _demo_poses['speedj_start'])

    current_color = [None]
    color_cyan = [0.0, 1.0, 0.8, 1.0]
    color_yellow = [1.0, 1.0, 0.0, 1.0]
    current_color[0] = color_cyan

    rec = TrailRecorder(backend, renderer,
                        trail_colors_fn=lambda: list(current_color[0]))
    rec.start()

    # Phase 1: speedj — joint velocity produces arc at EE (cyan)
    arm.speedj([0.3, 0.0, 0.0, -0.4, 0.0, 0.1, 0.0], acceleration=1.0, duration=3.0)

    # Phase 2: speedl — Cartesian velocity produces straight line (yellow)
    current_color[0] = color_yellow
    arm.speedl([0.0, 0.05, 0.0, 0.0, 0.0, 0.0], acceleration=0.5, duration=2.5)

    rec.stop()
    rec.hold_frame(0.5)

    save_gif(rec.frames, 'speedj.gif')
    arm.disable()
    renderer.close()


def demo_impedance():
    """Impedance control — compliant response to external perturbation."""
    print("Generating impedance.gif...")
    backend, renderer, arm = create_workstation()

    init_pose = _demo_poses['impedance_start']
    teleport_arm(arm, backend, init_pose)

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
            3.0 * math.sin(2 * math.pi * 0.3 * t), 0,
            3.0 * math.cos(2 * math.pi * 0.4 * t), 0, 0, 0]
        backend._data.xfrc_applied[link3_bid] = [
            0, 2.0 * math.sin(2 * math.pi * 0.5 * t), 0, 0, 0, 0]

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

    teleport_arm(arm, backend, _demo_poses['nullspace_start'])

    dynamics = arm._get_dynamics()
    fk = arm._get_fk()
    ik = arm._get_ik()
    dt = backend._control_dt
    n = arm.num_joints

    from onero_sdk.model_registry import get_model
    model_info = get_model("A1R")
    q_min = np.array([j.p_min for j in model_info.joints])
    q_max = np.array([j.p_max for j in model_info.joints])
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

    duration = 10.0
    steps = int(duration / dt)

    for step in range(steps):
        t = step * dt
        amp1 = 2.5 * math.sin(2 * math.pi * 0.15 * t)
        amp2 = 1.5 * math.sin(2 * math.pi * 0.25 * t + 1.5)
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
                cam_distance=1.7,
                cam_azimuth=-130,
                cam_elevation=-20,
                cam_lookat=_cam_params['lookat'],
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
            cam_distance=1.7,
            cam_azimuth=-130,
            cam_elevation=-20,
            cam_lookat=_cam_params['lookat'],
        )
        frames.append(img)

    save_gif(frames, 'nullspace.gif')
    arm.disable()
    renderer.close()


def demo_trajectory_profiles():
    """Trajectory profiles — trapezoidal velocity profile with large motion."""
    print("Generating trajectory_profiles.gif...")
    backend, renderer, arm = create_workstation()

    teleport_arm(arm, backend, _demo_poses['home'])

    rec = TrailRecorder(backend, renderer, trail_color=[0.0, 1.0, 0.0, 1.0])
    rec.start()
    arm.movej(_demo_poses['traj_target'], duration=2.5)
    rec.stop()
    rec.hold_frame(0.5)

    save_gif(rec.frames, 'trajectory_profiles.gif')
    arm.disable()
    renderer.close()


def demo_dual_arm():
    """Dual arm — mirrored motion through the public DualArm/Arm stack."""
    print("Generating dual_arm.gif...")
    from onero_sdk import DualArm
    from onero_sdk.safety import SafetyConfig
    from onero_sdk.dual_mujoco_backend import DualMujocoBackend

    backend = DualMujocoBackend(model_name="A1R", realtime=False, control_dt=0.005)
    left_backend, right_backend = backend.arm_backends()
    dual = DualArm.create(
        left_backend=left_backend,
        right_backend=right_backend,
        arm_spacing=0.24,
        # MuJoCo's simplified actuator tracking lags the SDK reference more
        # than hardware. Keep all kinematic/velocity/limit safety active while
        # disabling only the hardware tracking-error latch for this renderer.
        safety=SafetyConfig(
            tracking_error_hard_threshold=0,
            tracking_error_soft_threshold=0,
        ),
    )
    dual.enable()
    renderer = mujoco.Renderer(backend._model, height=HEIGHT, width=WIDTH)

    frames = []
    render_every = max(1, int(1.0 / (FPS * backend._control_dt)))
    recorded_qpos = []
    table_contact_cycles = []
    table_id = mujoco.mj_name2id(
        backend._model, mujoco.mjtObj.mjOBJ_BODY, "table")

    # DualArm executes its two Arm calls on worker threads. EGL contexts are
    # thread-affine, so record physics states in the control callback and render
    # them afterward on this main thread. Also fail generation if a demo target
    # ever pushes either arm into the table.
    def record_step(shared):
        recorded_qpos.append(shared._data.qpos[:14].copy())
        for index in range(shared._data.ncon):
            contact = shared._data.contact[index]
            body1 = shared._model.geom_bodyid[int(contact.geom1)]
            body2 = shared._model.geom_bodyid[int(contact.geom2)]
            if table_id in (body1, body2):
                table_contact_cycles.append(shared.control_cycles)
                break

    backend.on_step = record_step

    # With the canonical A1L/A1R URDFs and humanoid mounting matrices,
    # q_left = -q_right gives a world-space mirror about the sagittal Y=0 plane.
    right_poses = [
        [0.25, -0.10, 0.10, 0.30, 0.05, 0.20, 0.00],
        [-0.45, -0.20, 0.20, 0.55, 0.10, 0.30, 0.05],
        [0.00, 0.00, -0.30, 0.30, -0.20, 0.20, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    ]

    # Stream a smooth, synchronized trajectory through the public realtime
    # API. Unlike two independent blocking MoveJ planners, this guarantees one
    # left and one right command in every shared-scene control cycle.
    right_start = np.zeros(7)
    steps_per_segment = int(2.0 / backend._control_dt)
    for right_target in map(np.asarray, right_poses):
        delta = right_target - right_start
        for step in range(1, steps_per_segment + 1):
            u = step / steps_per_segment
            blend = u * u * (3.0 - 2.0 * u)
            right_command = right_start + blend * delta
            dual.servoj((-right_command).tolist(), right_command.tolist())
        right_start = right_target

    backend.on_step = None
    if table_contact_cycles:
        first_cycle = table_contact_cycles[0]
        raise RuntimeError(
            f"dual-arm demo contacted the table in "
            f"{len(table_contact_cycles)} control cycles "
            f"(first at cycle {first_cycle})")
    final_qpos = backend._data.qpos[:14].copy()
    for qpos in recorded_qpos[::render_every]:
        backend._data.qpos[:14] = qpos
        mujoco.mj_forward(backend._model, backend._data)
        frames.append(render_scene(
            renderer, backend._model, backend._data,
            cam_distance=2.0, cam_azimuth=-150,
            cam_elevation=-20, cam_lookat=[0, 0, 0.9]))
    backend._data.qpos[:14] = final_qpos
    mujoco.mj_forward(backend._model, backend._data)
    save_gif(frames, 'dual_arm.gif')
    dual.left.close()
    dual.right.close()
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
    print("  Generating README GIF animations (dual-arm scene)")
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
