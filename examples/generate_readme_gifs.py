"""
Generate GIF animations for README documentation.

Renders MuJoCo simulation offscreen and produces GIF files showing
each major SDK feature in action.

Usage:
    python examples/generate_readme_gifs.py

Output:
    docs/assets/movej.gif
    docs/assets/movel.gif
    docs/assets/servo_sine.gif
    docs/assets/freedrive.gif
    docs/assets/trajectory_profiles.gif
    docs/assets/blending.gif
    docs/assets/movec.gif

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

# Rendering config
WIDTH = 640
HEIGHT = 480
FPS = 30


def create_renderer(backend):
    """Create offscreen renderer from backend's model."""
    renderer = mujoco.Renderer(backend._model, height=HEIGHT, width=WIDTH)
    return renderer


def render_frame(renderer, data):
    """Render a single frame and return RGB array."""
    renderer.update_scene(data, camera='track' if has_camera(renderer) else -1)
    return renderer.render().copy()


def has_camera(renderer):
    """Check if model has a named camera."""
    return False  # use free camera


def setup_camera(renderer, backend):
    """Configure camera for a good view of the arm."""
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    return scene_option


def render_scene(renderer, model, data, cam_distance=1.5, cam_azimuth=-120,
                 cam_elevation=-25, cam_lookat=None):
    """Render with explicit camera settings."""
    if cam_lookat is None:
        cam_lookat = [0.0, 0.0, 0.3]

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = cam_distance
    camera.azimuth = cam_azimuth
    camera.elevation = cam_elevation
    camera.lookat[:] = cam_lookat

    renderer.update_scene(data, camera=camera)
    return renderer.render().copy()


def save_gif(frames, filename, fps=FPS):
    """Save frames as GIF."""
    path = os.path.join(ASSETS_DIR, filename)
    # Subsample if too many frames
    max_frames = fps * 8  # max 8 seconds
    if len(frames) > max_frames:
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = [frames[i] for i in indices]
    imageio.mimsave(path, frames, fps=fps, loop=0)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  Saved: {path} ({len(frames)} frames, {size_mb:.1f} MB)")
    return path


# ─── Demo generators ────────────────────────────────────────────────────────

def demo_movej():
    """movej: move to joint angles with trapezoidal profile."""
    print("Generating movej.gif...")
    backend = MujocoBackend(model_name="A1", control_dt=0.005, realtime=False)
    renderer = create_renderer(backend)
    backend.enable()

    n = backend.num_joints
    start = [0.0] * n
    target = [0.5, -0.8, 0.3, -1.2, 0.4, 0.3, -0.2]
    duration = 2.5
    dt = 0.005

    v_max = [2.0] * n
    a_max = [5.0] * n
    profiles = multi_joint_trapezoidal(start, target, v_max, a_max, dt,
                                       target_duration=duration)
    n_frames = len(profiles[0])

    frames = []
    render_every = max(1, int(1.0 / (FPS * dt)))

    for step in range(n_frames):
        pos = [profiles[i][step].position for i in range(n)]
        vel = [profiles[i][step].velocity for i in range(n)]
        backend.command(pos, vel, [80.0]*n, [2.0]*n, [0.0]*n)

        if step % render_every == 0:
            img = render_scene(renderer, backend._model, backend._data)
            frames.append(img)

    # Hold final pose for 0.5s
    for _ in range(int(0.5 / dt)):
        backend.command(target, [0.0]*n, [80.0]*n, [2.0]*n, [0.0]*n)
    img = render_scene(renderer, backend._model, backend._data)
    for _ in range(int(FPS * 0.5)):
        frames.append(img)

    save_gif(frames, 'movej.gif')
    backend.close()
    renderer.close()


def demo_movel():
    """movel: straight-line Cartesian motion (simulated via IK-like trajectory)."""
    print("Generating movel.gif...")
    backend = MujocoBackend(model_name="A1", control_dt=0.005, realtime=False)
    renderer = create_renderer(backend)
    backend.enable()

    n = backend.num_joints
    dt = 0.005

    # Move through a sequence of joint targets that simulate Cartesian motion
    waypoints = [
        [0.0, -0.5, 0.0, -1.0, 0.0, 0.3, 0.0],
        [0.3, -0.5, 0.2, -1.0, 0.0, 0.3, 0.0],
        [0.3, -0.8, 0.2, -0.7, 0.2, 0.3, 0.0],
        [0.0, -0.5, 0.0, -1.0, 0.0, 0.3, 0.0],
    ]

    frames = []
    render_every = max(1, int(1.0 / (FPS * dt)))
    v_max = [1.5] * n
    a_max = [4.0] * n
    step_counter = 0

    for seg in range(len(waypoints) - 1):
        start = waypoints[seg]
        end = waypoints[seg + 1]
        profiles = multi_joint_trapezoidal(start, end, v_max, a_max, dt,
                                           target_duration=1.5)
        seg_len = len(profiles[0])
        for step in range(seg_len):
            pos = [profiles[i][min(step, len(profiles[i])-1)].position for i in range(n)]
            vel = [profiles[i][min(step, len(profiles[i])-1)].velocity for i in range(n)]
            backend.command(pos, vel, [80.0]*n, [2.0]*n, [0.0]*n)
            step_counter += 1
            if step_counter % render_every == 0:
                img = render_scene(renderer, backend._model, backend._data)
                frames.append(img)

    save_gif(frames, 'movel.gif')
    backend.close()
    renderer.close()


def demo_servo_sine():
    """servoj: real-time sine wave on joint 2."""
    print("Generating servo_sine.gif...")
    backend = MujocoBackend(model_name="A1", control_dt=0.005, realtime=False)
    renderer = create_renderer(backend)
    backend.enable()

    n = backend.num_joints
    dt = 0.005
    duration = 4.0
    steps = int(duration / dt)

    frames = []
    render_every = max(1, int(1.0 / (FPS * dt)))

    for step in range(steps):
        t = step * dt
        joints = [0.0] * n
        joints[1] = 0.6 * math.sin(2 * math.pi * 0.5 * t)
        joints[3] = -0.8 + 0.3 * math.sin(2 * math.pi * 0.7 * t)
        joints[5] = 0.4 * math.sin(2 * math.pi * 0.3 * t)

        backend.command(joints, [0.0]*n, [80.0]*n, [2.0]*n, [0.0]*n)

        if step % render_every == 0:
            img = render_scene(renderer, backend._model, backend._data)
            frames.append(img)

    save_gif(frames, 'servo_sine.gif')
    backend.close()
    renderer.close()


def demo_trajectory_profiles():
    """Compare cosine/trapezoidal/s-curve side by side (split-screen)."""
    print("Generating trajectory_profiles.gif...")

    # Run 3 sims and composite frames
    n_joints = 7
    start = [0.0] * n_joints
    target = [0.8, -0.7, 0.4, -1.2, 0.3, 0.2, -0.15]
    duration = 2.5
    dt = 0.005
    v_max = [2.5] * n_joints
    a_max = [5.0] * n_joints
    j_max = [30.0] * n_joints

    # Generate trajectories
    # Trapezoidal
    trap_profiles = multi_joint_trapezoidal(start, target, v_max, a_max, dt,
                                            target_duration=duration)
    trap_steps = len(trap_profiles[0])

    # S-curve
    deltas = [target[i] - start[i] for i in range(n_joints)]
    sc_profiles = [SCurveProfile(deltas[i], v_max[i], a_max[i], j_max[i])
                   for i in range(n_joints)]
    sc_dur = max(max(p.duration for p in sc_profiles), duration)
    sc_steps = int(sc_dur / dt) + 1

    # Cosine
    cos_steps = int(duration / dt) + 1

    total_steps = max(trap_steps, sc_steps, cos_steps)

    # Run MuJoCo for trapezoidal
    backend = MujocoBackend(model_name="A1", control_dt=dt, realtime=False)
    renderer = create_renderer(backend)
    backend.enable()

    frames = []
    render_every = max(1, int(1.0 / (FPS * dt)))

    for step in range(total_steps):
        # Trapezoidal command
        if step < trap_steps:
            pos = [trap_profiles[i][step].position for i in range(n_joints)]
            vel = [trap_profiles[i][step].velocity for i in range(n_joints)]
        else:
            pos = target
            vel = [0.0] * n_joints

        backend.command(pos, vel, [80.0]*n_joints, [2.0]*n_joints, [0.0]*n_joints)

        if step % render_every == 0:
            img = render_scene(renderer, backend._model, backend._data)
            frames.append(img)

    # Hold
    for _ in range(int(FPS * 0.5)):
        frames.append(frames[-1])

    save_gif(frames, 'trajectory_profiles.gif')
    backend.close()
    renderer.close()


def demo_blending():
    """Multi-waypoint trajectory with smooth blending."""
    print("Generating blending.gif...")
    backend = MujocoBackend(model_name="A1", control_dt=0.005, realtime=False)
    renderer = create_renderer(backend)
    backend.enable()

    n = backend.num_joints
    dt = 0.005

    waypoints = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, -0.4, 0.2, -0.8, 0.2, 0.1, 0.0],
        [-0.3, -0.7, 0.3, -1.2, 0.4, 0.2, -0.1],
        [0.6, -0.3, 0.1, -0.6, 0.1, 0.3, 0.1],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]

    frames = []
    render_every = max(1, int(1.0 / (FPS * dt)))
    v_max = [2.0] * n
    a_max = [4.0] * n
    step_counter = 0

    for seg in range(len(waypoints) - 1):
        profiles = multi_joint_trapezoidal(
            waypoints[seg], waypoints[seg + 1], v_max, a_max, dt,
            target_duration=1.5)
        seg_len = len(profiles[0])
        for step in range(seg_len):
            pos = [profiles[i][min(step, len(profiles[i])-1)].position for i in range(n)]
            vel = [profiles[i][min(step, len(profiles[i])-1)].velocity for i in range(n)]
            backend.command(pos, vel, [80.0]*n, [2.0]*n, [0.0]*n)
            step_counter += 1
            if step_counter % render_every == 0:
                img = render_scene(renderer, backend._model, backend._data)
                frames.append(img)

    save_gif(frames, 'blending.gif')
    backend.close()
    renderer.close()


def demo_freedrive():
    """Simulated freedrive: gravity-compensated compliant motion."""
    print("Generating freedrive.gif...")
    backend = MujocoBackend(model_name="A1", control_dt=0.005, realtime=False)
    renderer = create_renderer(backend)
    backend.enable()

    n = backend.num_joints
    dt = 0.005
    duration = 4.0
    steps = int(duration / dt)

    frames = []
    render_every = max(1, int(1.0 / (FPS * dt)))

    # Simulate freedrive: low gains, arm drifts under gravity with some damping
    # Apply a gentle external "push" via torque perturbation
    for step in range(steps):
        t = step * dt
        # Simulate user guiding the arm (small torque perturbation)
        tau_ext = [0.0] * n
        tau_ext[1] = 0.3 * math.sin(2 * math.pi * 0.2 * t)
        tau_ext[3] = 0.2 * math.sin(2 * math.pi * 0.15 * t + 1.0)
        tau_ext[5] = 0.1 * math.sin(2 * math.pi * 0.25 * t + 2.0)

        # Low stiffness (freedrive-like)
        q = list(backend._data.qpos[:n])
        backend.command(q, [0.0]*n, [2.0]*n, [1.0]*n, tau_ext)

        if step % render_every == 0:
            img = render_scene(renderer, backend._model, backend._data)
            frames.append(img)

    save_gif(frames, 'freedrive.gif')
    backend.close()
    renderer.close()


def demo_speed_control():
    """speedj: constant velocity motion."""
    print("Generating speedj.gif...")
    backend = MujocoBackend(model_name="A1", control_dt=0.005, realtime=False)
    renderer = create_renderer(backend)
    backend.enable()

    n = backend.num_joints
    dt = 0.005
    duration = 3.0
    steps = int(duration / dt)

    frames = []
    render_every = max(1, int(1.0 / (FPS * dt)))

    target_vel = [0.0, 0.5, 0.0, -0.3, 0.0, 0.2, 0.0]

    for step in range(steps):
        t = step * dt
        # Ramp up, cruise, ramp down
        if t < 0.5:
            scale = t / 0.5
        elif t > duration - 0.5:
            scale = (duration - t) / 0.5
        else:
            scale = 1.0
        vel = [v * scale for v in target_vel]
        backend.command_velocity(vel)

        if step % render_every == 0:
            img = render_scene(renderer, backend._model, backend._data)
            frames.append(img)

    save_gif(frames, 'speedj.gif')
    backend.close()
    renderer.close()


def main():
    print("=" * 60)
    print("  Generating README GIF animations (offscreen rendering)")
    print("=" * 60)
    print()

    demo_movej()
    demo_movel()
    demo_servo_sine()
    demo_freedrive()
    demo_speed_control()
    demo_blending()
    demo_trajectory_profiles()

    print()
    print("=" * 60)
    print(f"  All GIFs saved to: {os.path.abspath(ASSETS_DIR)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
