"""
MuJoCo 轨迹验证 — 在动力学仿真中测试各 profile 的正确性。

功能:
    1. 对比 cosine / trapezoidal / S-curve 三种曲线的仿真表现
    2. 自动验证：终点到达精度、速度归零、跟踪误差范围
    3. 生成对比图（可选）

既作为 demo 演示，也作为回归测试工具。无参运行 = 交互模式（显示图）；
加 --check 运行 = CI 模式（仅打印结果 + 返回退出码）。

运行方式:
    python examples/mujoco_trajectory_comparison.py           # 交互模式
    python examples/mujoco_trajectory_comparison.py --check   # CI 验证模式

依赖:
    pip install -e ".[sim]"        # MuJoCo
    pip install matplotlib         # 绘图（交互模式）
"""
import sys
import math
import argparse
import numpy as np

sys.path.insert(0, '.')

from onero_sdk.mujoco_backend import MujocoBackend
from onero_sdk.trajectory_profiles import (
    TrapezoidalProfile, SCurveProfile, multi_joint_trapezoidal,
    compute_synchronized_duration,
)


# ─── Trajectory generators ─────────────────────────────────────────────────

def generate_cosine_trajectory(start, target, duration, dt):
    """Generate cosine interpolation frames (legacy profile)."""
    n = len(start)
    steps = max(1, int(duration / dt))
    positions = []
    velocities = []
    for step in range(steps + 1):
        t = step / steps
        alpha = 0.5 * (1.0 - math.cos(math.pi * t))
        pos = [start[i] + alpha * (target[i] - start[i]) for i in range(n)]
        dalpha = 0.5 * math.pi * math.sin(math.pi * t) / duration
        vel = [dalpha * (target[i] - start[i]) for i in range(n)]
        positions.append(pos)
        velocities.append(vel)
    return positions, velocities


def generate_trapezoidal_trajectory(start, target, v_max, a_max, dt, duration=None):
    """Generate trapezoidal profile frames."""
    profiles = multi_joint_trapezoidal(start, target, v_max, a_max, dt,
                                       target_duration=duration)
    n_frames = len(profiles[0])
    n_joints = len(start)
    positions = []
    velocities = []
    for step in range(n_frames):
        pos = [profiles[i][step].position for i in range(n_joints)]
        vel = [profiles[i][step].velocity for i in range(n_joints)]
        positions.append(pos)
        velocities.append(vel)
    return positions, velocities


def generate_scurve_trajectory(start, target, v_max, a_max, j_max, dt, duration=None):
    """Generate S-curve profile frames."""
    n = len(start)
    deltas = [target[i] - start[i] for i in range(n)]
    profiles = []
    for i in range(n):
        profiles.append(SCurveProfile(deltas[i], v_max[i], a_max[i], j_max[i]))

    s_dur = max(p.duration for p in profiles)
    if duration is not None:
        s_dur = max(s_dur, duration)

    n_frames = max(1, int(s_dur / dt)) + 1
    positions = []
    velocities = []
    for step in range(n_frames):
        t = step * dt
        pos = []
        vel = []
        for i in range(n):
            pt = profiles[i].evaluate(min(t, profiles[i].duration))
            pos.append(start[i] + pt.position)
            vel.append(pt.velocity)
        positions.append(pos)
        velocities.append(vel)
    return positions, velocities


# ─── MuJoCo execution ──────────────────────────────────────────────────────

def run_trajectory_in_mujoco(backend, start, positions, velocities, dt, kp=80.0, kd=2.0):
    """Execute a trajectory in MuJoCo and record actual states."""
    n = len(start)
    backend.reset(start)

    actual_pos = []
    actual_vel = []
    actual_torque = []
    cmd_pos = []

    for pos, vel in zip(positions, velocities):
        state = backend.command(
            positions=pos,
            velocities=vel,
            kps=[kp] * n,
            kds=[kd] * n,
            torques=[0.0] * n,
        )
        actual_pos.append(list(state.joints))
        actual_vel.append(list(state.velocities))
        actual_torque.append(list(state.torques))
        cmd_pos.append(pos)

    return {
        'cmd_pos': np.array(cmd_pos),
        'actual_pos': np.array(actual_pos),
        'actual_vel': np.array(actual_vel),
        'actual_torque': np.array(actual_torque),
    }


# ─── Validation checks ─────────────────────────────────────────────────────

def validate_trajectory(name, data, target, dt, tolerances):
    """
    Run automated checks on a trajectory execution result.

    Returns list of (check_name, passed, detail) tuples.
    """
    checks = []
    n_joints = len(target)

    # 1. Final position accuracy (settle within tolerance after motion + hold)
    final_pos = data['actual_pos'][-1]
    max_pos_err = max(abs(final_pos[i] - target[i]) for i in range(n_joints))
    passed = max_pos_err < tolerances['final_pos_rad']
    checks.append((
        'final_position',
        passed,
        f"max_err={max_pos_err*1000:.1f} mrad (limit: {tolerances['final_pos_rad']*1000:.0f} mrad)"
    ))

    # 2. Final velocity should be near zero (motion completed, not still moving)
    final_vel = data['actual_vel'][-1]
    max_final_vel = max(abs(v) for v in final_vel)
    passed = max_final_vel < tolerances['final_vel_rad_s']
    checks.append((
        'final_velocity_zero',
        passed,
        f"max_vel={max_final_vel*1000:.1f} mrad/s (limit: {tolerances['final_vel_rad_s']*1000:.0f} mrad/s)"
    ))

    # 3. No divergence — position should always be between start and target (+ margin)
    start_pos = data['cmd_pos'][0]
    overshoot = 0.0
    for i in range(n_joints):
        lo = min(start_pos[i], target[i]) - tolerances['overshoot_rad']
        hi = max(start_pos[i], target[i]) + tolerances['overshoot_rad']
        joint_actual = data['actual_pos'][:, i]
        under = np.min(joint_actual) - lo
        over = np.max(joint_actual) - hi
        overshoot = max(overshoot, max(0, -under), max(0, over))
    passed = overshoot < 1e-6
    checks.append((
        'no_overshoot',
        passed,
        f"overshoot={overshoot*1000:.1f} mrad (limit: {tolerances['overshoot_rad']*1000:.0f} mrad)"
    ))

    # 4. Monotonicity — command positions should move monotonically toward target
    cmd_pos = data['cmd_pos']
    monotonic = True
    for i in range(n_joints):
        delta = target[i] - start_pos[i]
        if abs(delta) < 1e-6:
            continue
        sign = 1 if delta > 0 else -1
        diffs = np.diff(cmd_pos[:, i]) * sign
        if np.any(diffs < -1e-10):
            monotonic = False
            break
    checks.append((
        'cmd_monotonic',
        monotonic,
        "command positions move monotonically toward target" if monotonic else "NON-MONOTONIC commands detected"
    ))

    # 5. Tracking error bound — should stay within acceptable range during motion
    tracking_err = np.abs(data['actual_pos'] - data['cmd_pos'])
    max_tracking = np.max(tracking_err)
    passed = max_tracking < tolerances['tracking_rad']
    checks.append((
        'tracking_error',
        passed,
        f"max={max_tracking*1000:.1f} mrad (limit: {tolerances['tracking_rad']*1000:.0f} mrad)"
    ))

    # 6. Torque bounds — should not saturate excessively
    max_torque = np.max(np.abs(data['actual_torque']))
    passed = max_torque < tolerances['max_torque_nm']
    checks.append((
        'torque_bounds',
        passed,
        f"peak={max_torque:.2f} Nm (limit: {tolerances['max_torque_nm']:.1f} Nm)"
    ))

    # 7. Smoothness — torque rate of change (jerk indicator)
    dtorque = np.diff(data['actual_torque'], axis=0) / dt
    max_dtorque = np.max(np.abs(dtorque))
    checks.append((
        'torque_smoothness',
        True,  # informational, no hard limit
        f"max dτ/dt = {max_dtorque:.1f} Nm/s"
    ))

    return checks


# ─── Plotting ───────────────────────────────────────────────────────────────

def plot_comparison(results, dt, joint_idx=1):
    """Plot comparison of different profiles for a single joint."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'Trajectory Profile Comparison — Joint {joint_idx+1}', fontsize=14)

    colors = {'cosine': '#2196F3', 'trapezoidal': '#4CAF50', 's_curve': '#FF9800'}

    # Position
    ax = axes[0]
    for name, data in results.items():
        t = np.arange(len(data['cmd_pos'])) * dt
        ax.plot(t, data['cmd_pos'][:, joint_idx], '--', color=colors[name],
                alpha=0.5, linewidth=1)
        ax.plot(t, data['actual_pos'][:, joint_idx], '-', color=colors[name],
                label=name, linewidth=1.5)
    ax.set_ylabel('Position (rad)')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_title('Command (dashed) vs Actual (solid)')
    ax.grid(True, alpha=0.3)

    # Tracking error
    ax = axes[1]
    for name, data in results.items():
        t = np.arange(len(data['cmd_pos'])) * dt
        error = (data['actual_pos'][:, joint_idx] - data['cmd_pos'][:, joint_idx]) * 1000
        ax.plot(t, error, '-', color=colors[name], label=name, linewidth=1.5)
    ax.set_ylabel('Tracking Error (mrad)')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title('Position Tracking Error')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linewidth=0.5)

    # Velocity
    ax = axes[2]
    for name, data in results.items():
        t = np.arange(len(data['actual_vel'])) * dt
        ax.plot(t, data['actual_vel'][:, joint_idx], '-', color=colors[name],
                label=name, linewidth=1.5)
    ax.set_ylabel('Velocity (rad/s)')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title('Actual Joint Velocity')
    ax.grid(True, alpha=0.3)

    # Torque
    ax = axes[3]
    for name, data in results.items():
        t = np.arange(len(data['actual_torque'])) * dt
        ax.plot(t, data['actual_torque'][:, joint_idx], '-', color=colors[name],
                label=name, linewidth=1.5)
    ax.set_ylabel('Torque (Nm)')
    ax.set_xlabel('Time (s)')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title('Control Torque (reflects acceleration smoothness)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_torque_rate(results, dt, joint_idx=1):
    """Plot torque derivative (jerk indicator)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    fig.suptitle(f'Torque Rate of Change (jerk indicator) — Joint {joint_idx+1}',
                 fontsize=13)

    colors = {'cosine': '#2196F3', 'trapezoidal': '#4CAF50', 's_curve': '#FF9800'}

    for name, data in results.items():
        torque = data['actual_torque'][:, joint_idx]
        dtorque = np.diff(torque) / dt
        t = np.arange(len(dtorque)) * dt
        ax.plot(t, dtorque, '-', color=colors[name], label=name,
                linewidth=1.2, alpha=0.8)

    ax.set_ylabel('d(torque)/dt (Nm/s)')
    ax.set_xlabel('Time (s)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linewidth=0.5)

    plt.tight_layout()
    return fig


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='MuJoCo trajectory profile validation')
    parser.add_argument('--check', action='store_true',
                        help='CI mode: validate and exit with code 0/1')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plotting even in interactive mode')
    args = parser.parse_args()

    # ─── Configuration ───
    n_joints = 7
    start = [0.0] * n_joints
    target = [0.8, -0.6, 0.4, -1.0, 0.3, 0.2, -0.15]
    duration = 2.0
    dt = 0.005  # 200Hz control

    v_max = [2.5] * n_joints
    a_max = [5.0] * n_joints
    j_max = [30.0] * n_joints

    # Tolerances for validation
    tolerances = {
        'final_pos_rad': 0.15,      # 150 mrad (PD tracking, no integrator)
        'final_vel_rad_s': 0.5,     # 500 mrad/s
        'overshoot_rad': 0.2,       # 200 mrad margin
        'tracking_rad': 0.2,        # 200 mrad max tracking error
        'max_torque_nm': 20.0,      # torque saturation limit
    }

    # Add hold phase: extra 0.5s at target to verify settling
    hold_steps = int(0.5 / dt)

    print("=" * 70)
    print("  MuJoCo Trajectory Validation")
    print("  Profiles: cosine / trapezoidal / S-curve")
    print("=" * 70)
    print()
    print(f"  Target:    {target}")
    print(f"  Duration:  {duration} s + 0.5 s hold")
    print(f"  Freq:      {1/dt:.0f} Hz")
    print(f"  Mode:      {'CI check' if args.check else 'Interactive'}")
    print()

    # Create MuJoCo backend
    print("Initializing MuJoCo simulation...")
    backend = MujocoBackend(model_name="A1R", control_dt=dt, realtime=False)
    backend.enable()
    print("Ready.\n")

    # ─── Generate and execute trajectories ───
    profiles_config = {
        'cosine': lambda: generate_cosine_trajectory(start, target, duration, dt),
        'trapezoidal': lambda: generate_trapezoidal_trajectory(
            start, target, v_max, a_max, dt, duration=duration),
        's_curve': lambda: generate_scurve_trajectory(
            start, target, v_max, a_max, j_max, dt, duration=duration),
    }

    results = {}
    all_checks = {}

    for name, gen_fn in profiles_config.items():
        print(f"Running {name}...")
        positions, velocities = gen_fn()

        # Append hold phase (stay at target with zero velocity)
        for _ in range(hold_steps):
            positions.append(list(target))
            velocities.append([0.0] * n_joints)

        data = run_trajectory_in_mujoco(backend, start, positions, velocities, dt)
        results[name] = data
        print(f"  Frames: {len(positions)} ({duration}s motion + 0.5s hold)")

        # Validate
        checks = validate_trajectory(name, data, target, dt, tolerances)
        all_checks[name] = checks

    backend.close()

    # ─── Print results ───
    print("\n" + "=" * 70)
    print("  VALIDATION RESULTS")
    print("=" * 70)

    all_passed = True
    for name, checks in all_checks.items():
        print(f"\n  [{name.upper()}]")
        for check_name, passed, detail in checks:
            status = "PASS" if passed else "FAIL"
            marker = " " if passed else "!"
            print(f"  {marker} [{status}] {check_name}: {detail}")
            if not passed:
                all_passed = False

    # Summary table
    print("\n" + "-" * 70)
    print(f"  {'Profile':<14} {'Final err':<14} {'Max track':<14} "
          f"{'Peak torque':<14} {'Settle vel':<14}")
    print("-" * 70)
    for name, data in results.items():
        final_err = max(abs(data['actual_pos'][-1, i] - target[i])
                       for i in range(n_joints))
        max_track = np.max(np.abs(data['actual_pos'] - data['cmd_pos']))
        peak_tau = np.max(np.abs(data['actual_torque']))
        final_vel = max(abs(v) for v in data['actual_vel'][-1])
        print(f"  {name:<14} {final_err*1000:>8.1f} mrad  {max_track*1000:>8.1f} mrad  "
              f"{peak_tau:>8.2f} Nm    {final_vel*1000:>8.1f} mrad/s")

    print()
    if all_passed:
        print("  ALL CHECKS PASSED")
    else:
        print("  SOME CHECKS FAILED — see details above")
    print("=" * 70)

    # ─── Plotting ───
    if not args.check and not args.no_plot:
        try:
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("\nmatplotlib not available — skipping plots.")
            print("Install with: pip install matplotlib")
            return 0 if all_passed else 1

        fig1 = plot_comparison(results, dt, joint_idx=1)
        fig2 = plot_torque_rate(results, dt, joint_idx=1)
        fig3 = plot_comparison(results, dt, joint_idx=3)

        plt.show()

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
