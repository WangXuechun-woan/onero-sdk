"""
Generate GIF animations for README documentation.

Renders MuJoCo simulation offscreen with end-effector trail visualization,
producing GIF files showing each major SDK feature in action.

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
from onero_sdk.mujoco_backend import MujocoBackend
from scipy.spatial.transform import Rotation

ASSETS_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'assets')
os.makedirs(ASSETS_DIR, exist_ok=True)

WIDTH = 640
HEIGHT = 480
FPS = 30


# ─── Rendering helpers ──────────────────────────────────────────────────────

def render_scene(renderer, model, data, trail_points=None,
                 cam_distance=1.5, cam_azimuth=-120, cam_elevation=-25,
                 cam_lookat=None, trail_color=None, trail_colors=None,
                 arrows=None):
    """Render with camera settings, trail visualization, and optional arrows."""
    if cam_lookat is None:
        cam_lookat = [0.0, 0.0, 0.3]
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


def get_ee_pos(data, body_id=7):
    """Get end-effector (Link7) world position."""
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
                 cam_distance=1.5, cam_azimuth=-120, cam_elevation=-25,
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
        self._cam_distance = cam_distance
        self._cam_azimuth = cam_azimuth
        self._cam_elevation = cam_elevation
        self._cam_lookat = cam_lookat

    def start(self):
        self.backend.command = self._hook_command
        self.backend.command_velocity = self._hook_velocity
        self.backend.command_torque = self._hook_torque

    def stop(self):
        self.backend.command = self._orig_command
        self.backend.command_velocity = self._orig_cmd_vel
        self.backend.command_torque = self._orig_cmd_torque

    def _record(self):
        self.trail.append(get_ee_pos(self.data))
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


# ─── Demo generators ────────────────────────────────────────────────────────

def demo_movej():
    """movej — joint-space motion with trapezoidal profile."""
    print("Generating movej.gif...")
    backend = MujocoBackend(model_name="A1", control_dt=0.005, realtime=False)
    renderer = mujoco.Renderer(backend._model, height=HEIGHT, width=WIDTH)
    arm = Arm("A1", backend=backend)
    arm.enable()

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
    backend = MujocoBackend(model_name="A1", control_dt=0.005, realtime=False)
    renderer = mujoco.Renderer(backend._model, height=HEIGHT, width=WIDTH)
    arm = Arm("A1", backend=backend)
    arm.enable()

    arm.movej([0.0, -0.5, 0.0, -0.8, 0.0, 0.3, 0.0], duration=1.0)

    start_ee = backend._data.xpos[7].copy()
    fk = arm._get_fk()
    _, start_rot = fk.get_ee_pose(arm.state.joints)
    start_rotvec = Rotation.from_matrix(start_rot).as_rotvec()

    # Large straight-line motion (~200mm)
    target_ee = start_ee + np.array([0.12, -0.06, 0.14])
    pose_6d = np.concatenate([target_ee, start_rotvec])

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
    backend = MujocoBackend(model_name="A1", control_dt=0.005, realtime=False)
    renderer = mujoco.Renderer(backend._model, height=HEIGHT, width=WIDTH)
    arm = Arm("A1", backend=backend)
    arm.enable()

    # Extended starting pose — larger reachable workspace for bigger arc
    arm.movej([0.0, -0.6, 0.0, -1.0, 0.0, 0.4, 0.0], duration=1.5)

    start_ee = backend._data.xpos[7].copy()
    fk = arm._get_fk()
    _, start_rot = fk.get_ee_pose(arm.state.joints)
    start_rotvec = Rotation.from_matrix(start_rot).as_rotvec()

    # Reset IK solver to current config for accurate tracking
    ik = arm._get_ik()
    ik.reset(np.array(arm.state.joints))

    # Large arc in XZ plane (~140mm chord, ~74mm radius)
    via_pos = start_ee + np.array([0.06, 0.0, 0.10])
    target_pos = start_ee + np.array([0.14, 0.0, 0.01])

    # Keep same rotation throughout (avoids IK divergence)
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
    """servoj — real-time streaming. Sine wave clearly visible in EE path.

    Drives joint 1 (shoulder) with sine to produce a clear vertical sine wave
    while slowly advancing joint 0 (base rotation) for horizontal sweep.
    """
    print("Generating servo_sine.gif...")
    backend = MujocoBackend(model_name="A1", control_dt=0.005, realtime=False)
    renderer = mujoco.Renderer(backend._model, height=HEIGHT, width=WIDTH)
    arm = Arm("A1", backend=backend)
    arm.enable()

    # Start at the exact first-frame pose so there's no initial jump
    base_pose = [-0.35, -0.6, 0.0, -1.0, 0.0, 0.4, 0.0]
    arm.movej(base_pose, duration=1.0)

    rec = TrailRecorder(backend, renderer, trail_color=[0.0, 0.5, 1.0, 1.0],
                        cam_distance=1.6, cam_azimuth=-180, cam_elevation=-10,
                        cam_lookat=[-0.3, 0.2, 0.35])
    rec.start()

    n = arm.num_joints
    duration = 5.0
    steps = int(duration / backend._control_dt)

    for step in range(steps):
        t = step * backend._control_dt
        joints = list(base_pose)
        # Slow base rotation for horizontal sweep (full range)
        joints[0] = 0.7 * (t / duration) - 0.35
        # Clear sine wave on shoulder → vertical oscillation (fewer cycles = clearer)
        joints[1] = base_pose[1] + 0.45 * math.sin(2 * math.pi * 0.8 * t)
        # Elbow counter-motion to amplify EE displacement
        joints[3] = base_pose[3] - 0.2 * math.sin(2 * math.pi * 0.8 * t)
        arm.servoj(joints)

    rec.stop()
    rec.hold_frame(0.8)

    save_gif(rec.frames, 'servo_sine.gif')
    arm.disable()
    renderer.close()


def demo_blending():
    """Multi-segment blending: movel → movec → movel with smooth transitions.

    Shows different trajectory types connected seamlessly — demonstrates
    the SDK's ability to blend between straight lines and arcs smoothly.
    """
    print("Generating blending.gif...")
    backend = MujocoBackend(model_name="A1", control_dt=0.005, realtime=False)
    renderer = mujoco.Renderer(backend._model, height=HEIGHT, width=WIDTH)
    arm = Arm("A1", backend=backend)
    arm.enable()

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

    # Segment 1: movel straight line up-right (green) ~130mm
    current_seg[0] = 0
    ee = backend._data.xpos[7].copy()
    target1 = ee + np.array([0.08, 0.0, 0.10])
    arm.movel(np.concatenate([target1, start_rotvec]), duration=2.0)

    # Segment 2: movec arc (orange) ~110mm chord
    current_seg[0] = 1
    ik.reset(np.array(arm.state.joints))
    ee = backend._data.xpos[7].copy()
    via_pos = ee + np.array([0.05, 0.02, 0.06])
    target2 = ee + np.array([0.11, 0.0, 0.0])
    arm.movec(
        np.concatenate([via_pos, start_rotvec]),
        np.concatenate([target2, start_rotvec]),
        duration=3.5,
    )

    # Segment 3: movel straight line in different direction (cyan) ~100mm
    current_seg[0] = 2
    ik.reset(np.array(arm.state.joints))
    ee = backend._data.xpos[7].copy()
    target3 = ee + np.array([0.0, -0.06, 0.08])
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
    """force_mode — hybrid force/position control.

    Phase 1: Push EE down (force in Z) — visible downward motion.
    Phase 2: Slide laterally while maintaining constant downward force.
    Shows force arrow indicator pointing in force direction.
    """
    print("Generating force_mode.gif...")
    backend = MujocoBackend(model_name="A1", control_dt=0.005, realtime=False)
    renderer = mujoco.Renderer(backend._model, height=HEIGHT, width=WIDTH)
    arm = Arm("A1", backend=backend)
    arm.enable()

    arm.movej([0.0, -0.4, 0.0, -0.6, 0.0, 0.2, 0.0], duration=1.5)

    # Force arrow state
    force_arrow = [None]

    def arrows_fn():
        if force_arrow[0] is not None:
            return [force_arrow[0]]
        return None

    rec = TrailRecorder(backend, renderer, trail_color=[1.0, 0.3, 0.0, 1.0],
                        arrows_fn=arrows_fn)
    rec.start()

    dt = backend._control_dt

    # Phase 1: Push down with large force (2.5s)
    steps_down = int(2.5 / dt)
    for step in range(steps_down):
        t = step * dt
        fz = -30.0 * min(1.0, t / 0.5)  # ramp to -30N over 0.5s
        arm.force_mode(
            selection=[0, 0, 1, 0, 0, 0],
            wrench=[0, 0, fz, 0, 0, 0],
            task_frame=[0, 0, 0, 0, 0, 0],
            limits=[0, 0, 0.25, 0, 0, 0],
        )
        ee_pos = backend._data.xpos[7].copy()
        arrow_len = abs(fz) * 0.004
        force_arrow[0] = {
            'from': ee_pos,
            'to': ee_pos + np.array([0, 0, -arrow_len]),
            'color': [1.0, 0.2, 0.2, 1.0],
        }

    # Phase 2: Slide in X while maintaining Z force (3.5s)
    steps_slide = int(3.5 / dt)
    for step in range(steps_slide):
        t = step * dt
        fx = 25.0  # constant lateral force
        fz = -30.0  # maintain downward force
        arm.force_mode(
            selection=[1, 0, 1, 0, 0, 0],
            wrench=[fx, 0, fz, 0, 0, 0],
            task_frame=[0, 0, 0, 0, 0, 0],
            limits=[0.20, 0, 0.25, 0, 0, 0],
        )
        ee_pos = backend._data.xpos[7].copy()
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
    """freedrive — teach mode + replay + 2x replay.

    Phase 1 (magenta): Simulated human dragging the arm (gravity-compensated).
    Phase 2 (green): Replay at 1x speed — same trajectory.
    Phase 3 (cyan): Replay at 2x speed — faster but same path.
    """
    print("Generating freedrive.gif...")
    backend = MujocoBackend(model_name="A1", control_dt=0.005, realtime=False)
    renderer = mujoco.Renderer(backend._model, height=HEIGHT, width=WIDTH)
    arm = Arm("A1", backend=backend)
    arm.enable()

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

    for step in range(steps):
        t = step * backend._control_dt
        # Gentle forces to produce smooth trajectory (stays within rate limits at 2x)
        backend._data.xfrc_applied[5] = [
            1.2 * math.sin(2 * math.pi * 0.2 * t),
            0.8 * math.sin(2 * math.pi * 0.15 * t),
            0.6 * math.cos(2 * math.pi * 0.12 * t), 0, 0, 0]
        backend._data.xfrc_applied[7] = [
            0.6 * math.sin(2 * math.pi * 0.25 * t + 2.0),
            0, 0.5 * math.cos(2 * math.pi * 0.18 * t), 0, 0, 0]

        q = list(backend._data.qpos[:n])
        g = arm.gravity_torque()
        arm.servoj(q, kp=0.0, kd=damping, tau_ff=g)

    backend._data.xfrc_applied[:] = 0
    recording = arm.stop_recording()

    # Fix timestamps: in non-realtime sim, wall-clock is compressed.
    # Rescale to actual simulation time (control_dt per frame).
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
    """speedj + speedl — joint velocity (arc) vs Cartesian velocity (straight line).

    Shows the difference: speedj produces an arc at EE (joint-space velocity),
    while speedl produces a straight line (Cartesian-space velocity).
    """
    print("Generating speedj.gif...")
    backend = MujocoBackend(model_name="A1", control_dt=0.005, realtime=False)
    renderer = mujoco.Renderer(backend._model, height=HEIGHT, width=WIDTH)
    arm = Arm("A1", backend=backend)
    arm.enable()

    # Track current segment color
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
    """Impedance control — compliant response to external perturbation.

    Shows the arm at a target pose resisting perturbation with low stiffness,
    creating visible spring-like deflection and recovery.
    Uses xfrc_applied (body-level external force) to simulate real perturbation.
    """
    print("Generating impedance.gif...")
    backend = MujocoBackend(model_name="A1", control_dt=0.005, realtime=False)
    renderer = mujoco.Renderer(backend._model, height=HEIGHT, width=WIDTH)
    arm = Arm("A1", backend=backend)
    arm.enable()

    init_pose = [0.3, -0.5, 0.2, -0.8, 0.2, 0.2, 0.0]
    arm.movej(init_pose, duration=1.5)

    rec = TrailRecorder(backend, renderer, trail_color=[0.7, 0.3, 1.0, 1.0])
    rec.start()

    n = arm.num_joints
    duration = 4.0
    steps = int(duration / backend._control_dt)

    for step in range(steps):
        t = step * backend._control_dt
        # External force on Link5 body (simulates human/object pushing)
        backend._data.xfrc_applied[5] = [
            5.0 * math.sin(2 * math.pi * 0.3 * t), 0,
            5.0 * math.cos(2 * math.pi * 0.4 * t), 0, 0, 0]
        backend._data.xfrc_applied[3] = [
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
    """Null-space motion — 7-DOF redundancy demonstration.

    The arm reconfigures dramatically (elbow moves ~220mm) while the
    end-effector stays perfectly fixed in space (<4mm drift). Uses the
    real IK solver to maintain EE constraint — same approach as on hardware.

    Green dot: EE (stays fixed). Magenta trail: elbow (moves freely).
    """
    print("Generating nullspace.gif...")
    backend = MujocoBackend(model_name="A1", control_dt=0.005, realtime=False)
    renderer = mujoco.Renderer(backend._model, height=HEIGHT, width=WIDTH)
    arm = Arm("A1", backend=backend)
    arm.enable()

    # More bent config gives larger null-space motion range
    arm.movej([0.0, -1.0, 0.0, -1.5, 0.0, 0.5, 0.0], duration=1.5)

    dynamics = arm._get_dynamics()
    fk = arm._get_fk()
    ik = arm._get_ik()
    dt = backend._control_dt
    n = arm.num_joints

    # Joint limits with safety margin
    q_min = np.array([-1.04, -3.14, -1.04, -3.14, -1.04, -3.14, -1.04])
    q_max = np.array([3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14])
    margin = 0.1

    # Lock EE target using FK (same as user would do)
    ee_target_pos, ee_target_rot = fk.get_ee_pose(list(arm.state.joints))

    # Multiple excitation directions for richer null-space exploration
    q_excite1 = np.array([0.4, -0.3, 1.2, 0.3, -1.0, 0.15, 0.6])
    q_excite2 = np.array([-0.3, 0.5, -0.6, 1.0, 0.4, -0.7, 0.3])
    q_current = np.array(arm.state.joints)
    ee_fixed_pos = get_ee_pos(backend._data, body_id=7)

    elbow_trail = []
    frames = []
    render_every = max(1, int(1.0 / (FPS * dt)))
    step_count = 0

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

        # IK solver corrects back to target EE pose
        ik.reset(q_candidate)
        q_current = np.array(ik.solve(ee_target_pos, ee_target_rot, dt=dt))

        arm.servoj(list(q_current), kp=100.0, kd=2.0)

        if step % 3 == 0:
            elbow_trail.append(backend._data.xpos[4].copy())

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
                cam_distance=1.3,
                cam_azimuth=-130,
                cam_elevation=-20,
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
            cam_distance=1.3,
            cam_azimuth=-130,
            cam_elevation=-20,
        )
        frames.append(img)

    save_gif(frames, 'nullspace.gif')
    arm.disable()
    renderer.close()


def demo_trajectory_profiles():
    """Trajectory profiles — trapezoidal velocity profile with large motion."""
    print("Generating trajectory_profiles.gif...")
    backend = MujocoBackend(model_name="A1", control_dt=0.005, realtime=False)
    renderer = mujoco.Renderer(backend._model, height=HEIGHT, width=WIDTH)
    arm = Arm("A1", backend=backend)
    arm.enable()

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

    backend = DualMujocoBackend(model_name="A1", realtime=False, control_dt=0.005)
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
    print("  Generating README GIF animations (offscreen rendering)")
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
