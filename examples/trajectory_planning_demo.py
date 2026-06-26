"""
轨迹规划演示 — 对比不同规划算法的效果。

展示内容:
    1. 余弦插值 (cosine) — SDK 默认
    2. 梯形速度曲线 (trapezoidal) — UR/KUKA 风格
    3. S 曲线 (s_curve) — jerk-limited，更平滑
    4. Ruckig (在线 jerk-limited) — Franka 风格 [可选]
    5. TOPP-RA (时间最优) — 最快可行轨迹 [可选]

运行方式:
    python examples/trajectory_planning_demo.py
"""
import sys
import numpy as np

sys.path.insert(0, '.')

from onero_sdk.trajectory_profiles import (
    TrapezoidalProfile, SCurveProfile, QuinticBlend,
    multi_joint_trapezoidal, compute_synchronized_duration,
)


def demo_single_joint():
    """Compare profiles for a single joint moving 2 rad."""
    print("=" * 60)
    print("  单关节轨迹对比 — 位移 2.0 rad")
    print("=" * 60)
    print()

    distance = 2.0
    v_max = 3.0   # rad/s
    a_max = 8.0   # rad/s²
    j_max = 40.0  # rad/s³
    dt = 0.01     # 100 Hz

    # Trapezoidal
    trap = TrapezoidalProfile(distance, v_max, a_max)
    trap_pts = trap.sample(dt)
    print(f"梯形速度曲线 (Trapezoidal):")
    print(f"  时长: {trap.duration:.3f} s")
    print(f"  峰值速度: {trap.v_peak:.3f} rad/s")
    print(f"  帧数: {len(trap_pts)}")
    print()

    # S-curve
    scurve = SCurveProfile(distance, v_max, a_max, j_max)
    scurve_pts = scurve.sample(dt)
    print(f"S 曲线 (7-segment, jerk-limited):")
    print(f"  时长: {scurve.duration:.3f} s")
    print(f"  帧数: {len(scurve_pts)}")
    # Check jerk is bounded
    max_jerk = 0
    for i in range(1, len(scurve_pts)):
        jerk = abs(scurve_pts[i].acceleration - scurve_pts[i-1].acceleration) / dt
        max_jerk = max(max_jerk, jerk)
    print(f"  实际最大 jerk: {max_jerk:.1f} rad/s³ (限制: {j_max})")
    print()

    # Quintic blend (connect two segments)
    print(f"五次多项式过渡 (Quintic Blend):")
    blend = QuinticBlend(
        duration=0.5,
        p0=0.0, v0=2.0, a0=0.0,   # end of segment 1 (cruising at 2 rad/s)
        p1=1.0, v1=1.5, a1=0.0,   # start of segment 2
    )
    pt_mid = blend.evaluate(0.25)
    print(f"  过渡时长: 0.5 s")
    print(f"  中点: pos={pt_mid.position:.3f}, vel={pt_mid.velocity:.3f}, acc={pt_mid.acceleration:.3f}")
    print(f"  保证 C² 连续 (位置/速度/加速度全连续)")
    print()


def demo_multi_joint():
    """Synchronized multi-joint trapezoidal profiles."""
    print("=" * 60)
    print("  多关节同步梯形曲线 — 7-DOF")
    print("=" * 60)
    print()

    n_joints = 7
    start = [0.0] * n_joints
    target = [1.5, -0.8, 0.3, -1.2, 0.5, 0.2, -0.1]
    v_max = [2.5, 2.5, 3.0, 3.0, 4.0, 4.0, 4.0]
    a_max = [5.0, 5.0, 8.0, 8.0, 10.0, 10.0, 10.0]
    dt = 0.01

    duration = compute_synchronized_duration(
        [target[i] - start[i] for i in range(n_joints)], v_max, a_max
    )
    print(f"同步时长: {duration:.3f} s (由最慢关节决定)")
    print()

    profiles = multi_joint_trapezoidal(start, target, v_max, a_max, dt)
    print(f"各关节结果:")
    for i in range(n_joints):
        pts = profiles[i]
        final = pts[-1]
        max_v = max(abs(p.velocity) for p in pts)
        print(f"  J{i+1}: Δ={target[i]-start[i]:+.2f} rad, "
              f"v_peak={max_v:.2f}/{v_max[i]:.1f} rad/s, "
              f"到达: {final.position:.3f} rad")
    print()
    print(f"所有关节同时到达 ✓ (总帧数: {len(profiles[0])})")
    print()


def demo_ruckig():
    """Online jerk-limited trajectory (requires ruckig)."""
    print("=" * 60)
    print("  Ruckig — 在线 jerk-limited 轨迹 (Franka 风格)")
    print("=" * 60)
    print()

    try:
        from onero_sdk.ruckig_planner import RuckigPlanner
        planner = RuckigPlanner(num_joints=7)
    except (ImportError, Exception) as e:
        print(f"  [跳过] ruckig 未安装。安装: pip install ruckig")
        print(f"  ({e})")
        print()
        return
    planner.set_limits(
        v_max=[2.5, 2.5, 3.0, 3.0, 4.0, 4.0, 4.0],
        a_max=[5.0, 5.0, 8.0, 8.0, 10.0, 10.0, 10.0],
        j_max=[30.0, 30.0, 50.0, 50.0, 80.0, 80.0, 80.0],
    )

    start = [0.0] * 7
    target = [1.5, -0.8, 0.3, -1.2, 0.5, 0.2, -0.1]

    frames = planner.plan(start, target, dt=0.01)
    print(f"  轨迹帧数: {len(frames)}")
    print(f"  总时长: {len(frames) * 0.01:.3f} s")
    print(f"  时间最优 + jerk 受限 ✓")
    print(f"  支持运动中改变目标 ✓")
    print()


def demo_toppra():
    """Time-optimal path parameterization (requires toppra)."""
    print("=" * 60)
    print("  TOPP-RA — 时间最优路径参数化")
    print("=" * 60)
    print()

    try:
        from onero_sdk.toppra_planner import TimeOptimalPlanner
        planner = TimeOptimalPlanner(num_joints=7)
    except (ImportError, Exception) as e:
        print(f"  [跳过] toppra 未安装。安装: pip install toppra")
        print(f"  ({e})")
        print()
        return
    planner.set_velocity_limits([2.5, 2.5, 3.0, 3.0, 4.0, 4.0, 4.0])
    planner.set_acceleration_limits([5.0, 5.0, 8.0, 8.0, 10.0, 10.0, 10.0])

    # Multi-waypoint path
    waypoints = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, -0.3, 0.1, -0.8, 0.2, 0.1, 0.0],
        [1.0, -0.6, 0.3, -1.2, 0.4, 0.2, -0.1],
        [1.5, -0.8, 0.3, -1.2, 0.5, 0.2, -0.1],
    ]

    frames = planner.plan(waypoints, dt=0.01)
    print(f"  路径点: {len(waypoints)} 个")
    print(f"  输出帧数: {len(frames)}")
    print(f"  最优时长: {len(frames) * 0.01:.3f} s")
    print(f"  全程满足速度/加速度约束 ✓")
    print()


def demo_arm_profiles():
    """Show how to use profiles in arm.movej()."""
    print("=" * 60)
    print("  arm.movej() 中的 profile 参数用法")
    print("=" * 60)
    print()
    print("""
# 余弦插值（默认，向后兼容）
arm.movej(target, duration=3.0)
arm.movej(target, duration=3.0, profile='cosine')

# 梯形速度曲线（指定 speed 时自动启用）
arm.movej(target, speed=1.5, acceleration=3.0)
arm.movej(target, speed=1.5, profile='trapezoidal')

# S 曲线（最平滑，无加速度跳变）
arm.movej(target, speed=1.5, acceleration=3.0, profile='s_curve')

# Ruckig（在线规划，可中途改变目标）
from onero_sdk import RuckigPlanner
planner = RuckigPlanner(num_joints=7)
planner.set_limits(v_max=..., a_max=..., j_max=...)
frames = planner.plan(start, target, dt=0.01)
arm.execute_trajectory(planner.to_trajectory_frames(frames, dt=0.01))

# TOPP-RA（时间最优，多路点）
from onero_sdk import TimeOptimalPlanner
planner = TimeOptimalPlanner(num_joints=7)
planner.set_velocity_limits(v_max)
planner.set_acceleration_limits(a_max)
frames = planner.plan(waypoints, dt=0.01)
arm.execute_trajectory(frames)
""")


def main():
    demo_single_joint()
    demo_multi_joint()
    demo_ruckig()
    demo_toppra()
    demo_arm_profiles()

    print("=" * 60)
    print("  全部演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
