"""
Generate GIF animations for README documentation.

Renders MuJoCo simulation offscreen with end-effector trail visualization,
producing GIF files showing each major SDK feature in action.

Usage:
    python examples/generate_readme_gifs.py           # generate all
    python examples/generate_readme_gifs.py movej     # generate specific demo

Output:
    docs/assets/*.gif

Dependencies:
    pip install -e ".[sim]"
    pip install imageio
"""
import sys
import os
import math
import numpy as np

# Use EGL for headless rendering (must be set before importing mujoco)
os.environ.setdefault('MUJOCO_GL', 'egl')

sys.path.insert(0, '.')

import mujoco
import imageio

from onero_sdk.mujoco_backend import MujocoBackend
from onero_sdk.trajectory_profiles import (
    TrapezoidalProfile, SCurveProfile, multi_joint_trapezoidal,
    compute_synchronized_duration,
)

ASSETS_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'assets')
os.makedirs(ASSETS_DIR, exist_ok=True)

WIDTH = 640
HEIGHT = 480
FPS = 30


# ─── Rendering helpers ──────────────────────────────────────────────────────

def render_scene(renderer, model, data, trail_points=None,
                 cam_distance=1.5, cam_azimuth=-120, cam_elevation=-25,
                 cam_lookat=None, trail_color=None, trail_colors=None):
    """Render with camera settings and optional trail visualization.

    trail_colors: per-point RGBA list (overrides trail_color if provided).
    """
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

    # Add trail spheres to scene
    if trail_points and len(trail_points) > 0:
        scene = renderer._scene
        max_trail = min(len(trail_points), scene.maxgeom - scene.ngeom - 10)
        # Subsample if too many points
        if len(trail_points) > max_trail:
            indices = np.linspace(0, len(trail_points) - 1, max_trail, dtype=int)
            points = [trail_points[i] for i in indices]
            colors = [trail_colors[i] for i in indices] if trail_colors else None
        else:
            points = trail_points
            colors = trail_colors

        for k, pt in enumerate(points):
            if scene.ngeom >= scene.maxgeom - 1:
                break
            idx = scene.ngeom
            scene.ngeom += 1
            color = colors[k] if colors else trail_color
            mujoco.mjv_initGeom(
                scene.geoms[idx],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.012, 0.012, 0.012],
                pos=np.asarray(pt, dtype=np.float64),
                mat=np.eye(3).flatten(),
                rgba=np.asarray(color, dtype=np.float32),
            )
            scene.geoms[idx].emission = 1.0

    return renderer.render().copy()


def get_ee_pos(data, body_id=7):
    """Get end-effector (Link7) world position."""
    return data.xpos[body_id].copy()


def save_gif(frames, filename, fps=FPS):
    """Save frames as GIF with optimization."""
    path = os.path.join(ASSETS_DIR, filename)
    # Cap at 6 seconds
    max_frames = fps * 6
    if len(frames) > max_frames:
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = [frames[i] for i in indices]
    imageio.mimsave(path, frames, fps=fps, loop=0)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  Saved: {path} ({len(frames)} frames, {size_mb:.1f} MB)")
    return path


def run_profiles(backend, start, target, dt, duration=None):
    """Helper: generate and return multi-joint trapezoidal profiles."""
    n = len(start)
    v_max = [2.0] * n
    a_max = [5.0] * n
    return multi_joint_trapezoidal(start, target, v_max, a_max, dt,
                                   target_duration=duration)


# ─── Demo generators ────────────────────────────────────────────────────────

def demo_movej():
    """movej: move to joint angles with trail."""
    print("Generating movej.gif...")
    backend = MujocoBackend(model_name="A1", control_dt=0.005, realtime=False)
    renderer = mujoco.Renderer(backend._model, height=HEIGHT, width=WIDTH)
    backend.enable()

    n = backend.num_joints
    dt = 0.005
    start = [0.0] * n
    target = [0.5, -0.8, 0.3, -1.2, 0.4, 0.3, -0.2]

    profiles = run_profiles(backend, start, target, dt, duration=2.5)
    n_frames = len(profiles[0])

    frames = []
    trail = []
    render_every = max(1, int(1.0 / (FPS * dt)))

    for step in range(n_frames):
        pos = [profiles[i][min(step, len(profiles[i])-1)].position for i in range(n)]
        vel = [profiles[i][min(step, len(profiles[i])-1)].velocity for i in range(n)]
        backend.command(pos, vel, [80.0]*n, [2.0]*n, [0.0]*n)
        trail.append(get_ee_pos(backend._data))

        if step % render_every == 0:
            img = render_scene(renderer, backend._model, backend._data,
                               trail_points=trail)
            frames.append(img)

    # Hold
    for _ in range(int(0.5 / dt)):
        backend.command(target, [0.0]*n, [80.0]*n, [2.0]*n, [0.0]*n)
    for _ in range(int(FPS * 0.5)):
        img = render_scene(renderer, backend._model, backend._data,
                           trail_points=trail)
        frames.append(img)

    save_gif(frames, 'movej.gif')
    backend.close()
    renderer.close()


def demo_movel():
    """movel: straight-line motion via SDK Arm.movel (Cartesian IK)."""
    print("Generating movel.gif...")
    backend = MujocoBackend(model_name="A1", control_dt=0.005, realtime=False)
    renderer = mujoco.Renderer(backend._model, height=HEIGHT, width=WIDTH)

    from onero_sdk import Arm
    arm = Arm("A1", backend=backend)
    arm.enable()

    model = backend._model
    data = backend._data
    n = backend.num_joints
    ee_body = 7

    # Move to a start pose
    arm.movej([0.2, -0.5, 0.0, -0.8, 0.0, 0.2, 0.0], duration=1.0)

    # Get start EE position
    start_ee = data.xpos[ee_body].copy()

    # Target: move ~12cm diagonally (produces a clear straight line)
    target_ee = start_ee + np.array([0.08, -0.05, 0.08])
    pose_6d = np.concatenate([target_ee, [0, 0, 0]])

    # Intercept command calls to collect trail + render frames
    frames = []
    trail = []
    render_every = max(1, int(1.0 / (FPS * 0.005)))
    step_counter = [0]
    original_command = backend.command

    def tracking_command(*args, **kwargs):
        result = original_command(*args, **kwargs)
        trail.append(data.xpos[ee_body].copy())
        step_counter[0] += 1
        if step_counter[0] % render_every == 0:
            img = render_scene(renderer, model, data,
                               trail_points=trail,
                               trail_color=[0.0, 1.0, 0.0, 1.0])
            frames.append(img)
        return result

    backend.command = tracking_command
    arm.movel(position=pose_6d, duration=2.5)
    backend.command = original_command

    # Hold with full trail visible
    for _ in range(int(FPS * 1.0)):
        img = render_scene(renderer, model, data,
                           trail_points=trail,
                           trail_color=[0.0, 1.0, 0.0, 1.0])
        frames.append(img)

    save_gif(frames, 'movel.gif')
    arm.disable()
    renderer.close()


def demo_servo_sine():
    """servoj: sine wave with trail showing the sinusoidal pattern."""
    print("Generating servo_sine.gif...")
    backend = MujocoBackend(model_name="A1", control_dt=0.005, realtime=False)
    renderer = mujoco.Renderer(backend._model, height=HEIGHT, width=WIDTH)
    backend.enable()

    n = backend.num_joints
    dt = 0.005
    duration = 4.0
    steps = int(duration / dt)

    frames = []
    trail = []
    render_every = max(1, int(1.0 / (FPS * dt)))

    for step in range(steps):
        t = step * dt
        joints = [0.0] * n
        joints[1] = 0.6 * math.sin(2 * math.pi * 0.5 * t)
        joints[3] = -0.8 + 0.3 * math.sin(2 * math.pi * 0.7 * t)
        joints[5] = 0.4 * math.sin(2 * math.pi * 0.3 * t)

        backend.command(joints, [0.0]*n, [80.0]*n, [2.0]*n, [0.0]*n)
        trail.append(get_ee_pos(backend._data))

        if step % render_every == 0:
            img = render_scene(renderer, backend._model, backend._data,
                               trail_points=trail,
                               trail_color=[0.0, 0.5, 1.0, 1.0])
            frames.append(img)

    save_gif(frames, 'servo_sine.gif')
    backend.close()
    renderer.close()


def demo_movec():
    """movec: circular arc motion with trail clearly showing the arc."""
    print("Generating movec.gif...")
    backend = MujocoBackend(model_name="A1", control_dt=0.005, realtime=False)
    renderer = mujoco.Renderer(backend._model, height=HEIGHT, width=WIDTH)
    backend.enable()

    n = backend.num_joints
    dt = 0.005

    # Generate a circular trajectory in joint space that produces
    # a visible arc in Cartesian space
    center = [0.3, -0.6, 0.2, -0.9, 0.2, 0.25, 0.0]
    amplitude = [0.25, 0.15, 0.12, 0.2, 0.12, 0.08, 0.04]

    duration = 4.0
    steps = int(duration / dt)
    frames = []
    trail = []
    render_every = max(1, int(1.0 / (FPS * dt)))

    for step in range(steps):
        t = step * dt
        angle = 2 * math.pi * t / duration
        pos = [center[i] + amplitude[i] * math.sin(angle + i * 0.5)
               for i in range(n)]
        vel = [amplitude[i] * (2 * math.pi / duration) *
               math.cos(angle + i * 0.5) for i in range(n)]
        backend.command(pos, vel, [80.0]*n, [2.0]*n, [0.0]*n)
        trail.append(get_ee_pos(backend._data))

        if step % render_every == 0:
            img = render_scene(renderer, backend._model, backend._data,
                               trail_points=trail,
                               trail_color=[1.0, 0.7, 0.0, 1.0])
            frames.append(img)

    save_gif(frames, 'movec.gif')
    backend.close()
    renderer.close()


def demo_freedrive():
    """Freedrive: compliant gravity-compensated motion."""
    print("Generating freedrive.gif...")
    backend = MujocoBackend(model_name="A1", control_dt=0.005, realtime=False)
    renderer = mujoco.Renderer(backend._model, height=HEIGHT, width=WIDTH)
    backend.enable()

    n = backend.num_joints
    dt = 0.005
    duration = 4.0
    steps = int(duration / dt)

    frames = []
    trail = []
    render_every = max(1, int(1.0 / (FPS * dt)))

    for step in range(steps):
        t = step * dt
        tau_ext = [0.0] * n
        tau_ext[1] = 0.4 * math.sin(2 * math.pi * 0.2 * t)
        tau_ext[3] = 0.3 * math.sin(2 * math.pi * 0.15 * t + 1.0)
        tau_ext[5] = 0.15 * math.sin(2 * math.pi * 0.25 * t + 2.0)

        q = list(backend._data.qpos[:n])
        backend.command(q, [0.0]*n, [2.0]*n, [1.0]*n, tau_ext)
        trail.append(get_ee_pos(backend._data))

        if step % render_every == 0:
            img = render_scene(renderer, backend._model, backend._data,
                               trail_points=trail,
                               trail_color=[1.0, 0.3, 1.0, 1.0])
            frames.append(img)

    save_gif(frames, 'freedrive.gif')
    backend.close()
    renderer.close()


def demo_speed_control():
    """speedj: constant velocity with trail."""
    print("Generating speedj.gif...")
    backend = MujocoBackend(model_name="A1", control_dt=0.005, realtime=False)
    renderer = mujoco.Renderer(backend._model, height=HEIGHT, width=WIDTH)
    backend.enable()

    n = backend.num_joints
    dt = 0.005
    duration = 3.0
    steps = int(duration / dt)

    frames = []
    trail = []
    render_every = max(1, int(1.0 / (FPS * dt)))

    target_vel = [0.0, 0.5, 0.0, -0.3, 0.0, 0.2, 0.0]

    for step in range(steps):
        t = step * dt
        if t < 0.5:
            scale = t / 0.5
        elif t > duration - 0.5:
            scale = (duration - t) / 0.5
        else:
            scale = 1.0
        vel = [v * scale for v in target_vel]
        backend.command_velocity(vel)
        trail.append(get_ee_pos(backend._data))

        if step % render_every == 0:
            img = render_scene(renderer, backend._model, backend._data,
                               trail_points=trail,
                               trail_color=[0.0, 1.0, 0.8, 1.0])
            frames.append(img)

    save_gif(frames, 'speedj.gif')
    backend.close()
    renderer.close()


def demo_blending():
    """Multi-segment blending: line + arc + curve with color-coded trail."""
    print("Generating blending.gif...")
    backend = MujocoBackend(model_name="A1", control_dt=0.005, realtime=False)
    renderer = mujoco.Renderer(backend._model, height=HEIGHT, width=WIDTH)
    backend.enable()

    n = backend.num_joints
    dt = 0.005

    # Waypoints designed to produce distinct EE path shapes:
    # Seg 1 (green):  straight reach forward
    # Seg 2 (orange): sweep right (arc in EE space)
    # Seg 3 (blue):   sweep left (arc)
    # Seg 4 (pink):   curved return
    # Seg 5 (gray):   home
    waypoints = [
        [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
        [0.0, -0.6,  0.0, -0.4,  0.0,  0.2,  0.0],
        [0.4, -0.8,  0.3, -1.0,  0.3,  0.3,  0.0],
        [-0.3, -0.5, 0.2, -1.2,  0.5,  0.4, -0.1],
        [0.2, -0.3,  0.1, -0.5,  0.1,  0.1,  0.0],
        [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
    ]

    seg_colors = [
        [0.0, 1.0, 0.0, 1.0],   # neon green (line)
        [1.0, 0.5, 0.0, 1.0],   # bright orange (arc)
        [0.0, 0.8, 1.0, 1.0],   # cyan (arc)
        [1.0, 0.0, 1.0, 1.0],   # magenta (curve)
        [1.0, 1.0, 0.0, 1.0],   # yellow (return)
    ]

    frames = []
    trail = []
    trail_colors = []
    render_every = max(1, int(1.0 / (FPS * dt)))
    step_counter = 0
    v_max = [1.8] * n
    a_max = [4.0] * n

    for seg in range(len(waypoints) - 1):
        profiles = multi_joint_trapezoidal(
            waypoints[seg], waypoints[seg + 1], v_max, a_max, dt,
            target_duration=1.5)
        seg_len = len(profiles[0])
        color = seg_colors[seg]
        for step in range(seg_len):
            pos = [profiles[i][min(step, len(profiles[i])-1)].position for i in range(n)]
            vel = [profiles[i][min(step, len(profiles[i])-1)].velocity for i in range(n)]
            backend.command(pos, vel, [150.0]*n, [5.0]*n, [0.0]*n)
            trail.append(get_ee_pos(backend._data))
            trail_colors.append(color)
            step_counter += 1
            if step_counter % render_every == 0:
                img = render_scene(renderer, backend._model, backend._data,
                                   trail_points=trail,
                                   trail_colors=trail_colors)
                frames.append(img)

    # Hold final with full trail visible
    for _ in range(int(FPS * 1.0)):
        img = render_scene(renderer, backend._model, backend._data,
                           trail_points=trail,
                           trail_colors=trail_colors)
        frames.append(img)

    save_gif(frames, 'blending.gif')
    backend.close()
    renderer.close()


def demo_trajectory_profiles():
    """Trapezoidal profile with trail."""
    print("Generating trajectory_profiles.gif...")
    backend = MujocoBackend(model_name="A1", control_dt=0.005, realtime=False)
    renderer = mujoco.Renderer(backend._model, height=HEIGHT, width=WIDTH)
    backend.enable()

    n = backend.num_joints
    dt = 0.005
    start = [0.0] * n
    target = [0.8, -0.7, 0.4, -1.2, 0.3, 0.2, -0.15]
    v_max = [2.5] * n
    a_max = [5.0] * n

    profiles = multi_joint_trapezoidal(start, target, v_max, a_max, dt,
                                       target_duration=2.5)
    n_steps = len(profiles[0])

    frames = []
    trail = []
    render_every = max(1, int(1.0 / (FPS * dt)))

    for step in range(n_steps):
        pos = [profiles[i][min(step, len(profiles[i])-1)].position for i in range(n)]
        vel = [profiles[i][min(step, len(profiles[i])-1)].velocity for i in range(n)]
        backend.command(pos, vel, [80.0]*n, [2.0]*n, [0.0]*n)
        trail.append(get_ee_pos(backend._data))

        if step % render_every == 0:
            img = render_scene(renderer, backend._model, backend._data,
                               trail_points=trail,
                               trail_color=[0.0, 1.0, 0.0, 1.0])
            frames.append(img)

    # Hold
    for _ in range(int(FPS * 0.5)):
        img = render_scene(renderer, backend._model, backend._data,
                           trail_points=trail,
                           trail_color=[0.0, 1.0, 0.0, 1.0])
        frames.append(img)

    save_gif(frames, 'trajectory_profiles.gif')
    backend.close()
    renderer.close()


def demo_force_mode():
    """Force mode: compliant pressing motion."""
    print("Generating force_mode.gif...")
    backend = MujocoBackend(model_name="A1", control_dt=0.005, realtime=False)
    renderer = mujoco.Renderer(backend._model, height=HEIGHT, width=WIDTH)
    backend.enable()

    n = backend.num_joints
    dt = 0.005

    # Phase 1: reach pose
    start = [0.0] * n
    reach = [0.2, -0.7, 0.1, -1.0, 0.0, 0.4, 0.0]
    profiles = run_profiles(backend, start, reach, dt, duration=1.5)

    frames = []
    trail = []
    render_every = max(1, int(1.0 / (FPS * dt)))

    for step in range(len(profiles[0])):
        pos = [profiles[i][min(step, len(profiles[i])-1)].position for i in range(n)]
        vel = [profiles[i][min(step, len(profiles[i])-1)].velocity for i in range(n)]
        backend.command(pos, vel, [80.0]*n, [2.0]*n, [0.0]*n)
        trail.append(get_ee_pos(backend._data))
        if step % render_every == 0:
            img = render_scene(renderer, backend._model, backend._data,
                               trail_points=trail,
                               trail_color=[1.0, 0.7, 0.0, 1.0])
            frames.append(img)

    # Phase 2: force control (compliant push)
    force_steps = int(2.0 / dt)
    for step in range(force_steps):
        t = step * dt
        target = list(reach)
        target[1] = reach[1] - 0.12 * (1 - math.exp(-t * 2.0))
        target[3] = reach[3] - 0.08 * (1 - math.exp(-t * 2.0))
        kps = [80.0] * n
        kps[1] = 20.0
        kps[3] = 20.0
        backend.command(target, [0.0]*n, kps, [2.0]*n, [0.0]*n)
        trail.append(get_ee_pos(backend._data))
        if step % render_every == 0:
            img = render_scene(renderer, backend._model, backend._data,
                               trail_points=trail,
                               trail_color=[1.0, 0.7, 0.0, 1.0])
            frames.append(img)

    save_gif(frames, 'force_mode.gif')
    backend.close()
    renderer.close()


def demo_dual_arm():
    """Dual arm: mirror motion with both arms."""
    print("Generating dual_arm.gif...")
    from onero_sdk.dual_mujoco_backend import DualMujocoBackend

    backend = DualMujocoBackend(model_name="A1", realtime=False, control_dt=0.005)
    renderer = mujoco.Renderer(backend._model, height=HEIGHT, width=WIDTH)

    dt = 0.005
    frames = []
    render_every = max(1, int(1.0 / (FPS * dt)))
    step_counter = 0

    poses = [
        ([1.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0]),
        ([0.5, -0.5, 0.3, -1.0, 0.2, 0.2, 0.0], [-0.5, 0.5, -0.3, 1.0, -0.2, -0.2, 0.0]),
        ([1.2, -0.3, 0.0, -0.8, 0.4, 0.0, 0.0], [-1.2, 0.3, 0.0, 0.8, -0.4, 0.0, 0.0]),
        ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ]

    for r_target, l_target in poses:
        r_start = list(backend._data.qpos[:7])
        l_start = list(backend._data.qpos[7:14])
        n_steps = int(1.5 / dt)

        for i in range(n_steps):
            alpha = (i + 1) / n_steps
            r_cmd = [r_start[j] + alpha * (r_target[j] - r_start[j]) for j in range(7)]
            l_cmd = [l_start[j] + alpha * (l_target[j] - l_start[j]) for j in range(7)]
            backend.step(r_cmd, l_cmd, kp=150.0, kd=5.0)
            step_counter += 1

            if step_counter % render_every == 0:
                img = render_scene(renderer, backend._model, backend._data,
                                   cam_distance=2.0, cam_azimuth=-150,
                                   cam_elevation=-20, cam_lookat=[0, 0, 0.9])
                frames.append(img)

    save_gif(frames, 'dual_arm.gif')
    renderer.close()


def demo_impedance():
    """Impedance control: compliant response to perturbation."""
    print("Generating impedance.gif...")
    backend = MujocoBackend(model_name="A1", control_dt=0.005, realtime=False)
    renderer = mujoco.Renderer(backend._model, height=HEIGHT, width=WIDTH)
    backend.enable()

    n = backend.num_joints
    dt = 0.005

    init_pose = [0.3, -0.5, 0.2, -0.8, 0.2, 0.2, 0.0]
    profiles = run_profiles(backend, [0.0]*n, init_pose, dt, duration=1.0)

    frames = []
    trail = []
    render_every = max(1, int(1.0 / (FPS * dt)))

    for step in range(len(profiles[0])):
        pos = [profiles[i][min(step, len(profiles[i])-1)].position for i in range(n)]
        vel = [profiles[i][min(step, len(profiles[i])-1)].velocity for i in range(n)]
        backend.command(pos, vel, [80.0]*n, [2.0]*n, [0.0]*n)

    # Impedance: low stiffness + perturbation
    duration = 3.5
    steps = int(duration / dt)
    for step in range(steps):
        t = step * dt
        tau_ext = [0.0] * n
        tau_ext[1] = 1.5 * math.sin(2 * math.pi * 0.5 * t)
        tau_ext[3] = 1.0 * math.sin(2 * math.pi * 0.7 * t + 0.5)
        tau_ext[5] = 0.5 * math.cos(2 * math.pi * 0.3 * t)

        backend.command(init_pose, [0.0]*n, [15.0]*n, [1.5]*n, tau_ext)
        trail.append(get_ee_pos(backend._data))

        if step % render_every == 0:
            img = render_scene(renderer, backend._model, backend._data,
                               trail_points=trail,
                               trail_color=[0.7, 0.3, 1.0, 1.0])
            frames.append(img)

    save_gif(frames, 'impedance.gif')
    backend.close()
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
}


def main():
    print("=" * 60)
    print("  Generating README GIF animations (offscreen rendering)")
    print("=" * 60)
    print()

    # Allow selecting specific demos via CLI args
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
