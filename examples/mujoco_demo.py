"""
MuJoCo 可视化演示 — onero SDK 全功能展示

功能列表:
  1. movej      — 关节空间运动（多 waypoint）
  2. movel      — 笛卡尔直线运动
  3. movec      — 圆弧运动
  4. servoj     — 实时关节伺服（正弦波）
  5. speedj     — 关节速度控制
  6. speedl     — 笛卡尔速度控制
  7. blending   — 多 waypoint 轨迹融合
  8. record+replay — 轨迹录制与回放
  9. freedrive  — 零重力示教模式
  10. force_mode — 力/位混合控制
  11. stiffness  — 刚度预设切换对比
  12. wrench     — 外力估计
  13. collision  — 碰撞检测

运行方式:
  conda activate onerosdk
  python examples/mujoco_demo.py

按 Ctrl+C 可随时退出。
"""

import sys
import os
import time
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mujoco
import mujoco.viewer

from onero_sdk import Arm, ArmState, TrajectoryPlanner, Waypoint
from onero_sdk.mujoco_backend import MujocoBackend
from onero_sdk.gain_policy import Stiffness
from onero_sdk.ik_solver import IKSolver


# ─── Helpers ─────────────────────────────────────────────────────────────────

def setup_scene(model):
    """添加地面、灯光等环境元素，改善渲染效果。"""
    # 地面格纹
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 720
    # 背景色（浅灰蓝天空）
    model.vis.rgba.fog[:] = [0.8, 0.85, 0.9, 1.0]
    # 头顶灯光增强
    model.vis.headlight.ambient[:] = [0.3, 0.3, 0.3]
    model.vis.headlight.diffuse[:] = [0.6, 0.6, 0.6]
    model.vis.headlight.specular[:] = [0.2, 0.2, 0.2]


def make_viewer_backend(model_name="A1"):
    """创建 MuJoCo 后端并启动被动渲染器（含环境改善）。

    通过 on_step 回调实现运动过程自动刷新渲染器，
    所有阻塞式 API（movej/movel/speedj 等）都会平滑显示。
    """
    backend = MujocoBackend(model_name=model_name, realtime=True)
    setup_scene(backend._model)
    viewer = mujoco.viewer.launch_passive(backend._model, backend._data)
    backend._on_step = lambda: viewer.sync()
    return backend, viewer


def wait_viewer(viewer, seconds):
    """等待指定时间，保持渲染器刷新。"""
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        viewer.sync()
        time.sleep(0.016)


def print_header(num, title, subtitle=""):
    print(f"\n{'─'*60}")
    print(f"  Demo {num}: {title}")
    if subtitle:
        print(f"  {subtitle}")
    print(f"{'─'*60}")


def go_home(arm, duration=2.0):
    """回零位。"""
    arm.movej([0.0] * 7, duration=duration)


# ─── Demo 1: movej ───────────────────────────────────────────────────────────

def demo_movej(arm, viewer):
    """关节空间运动 — 多目标点平滑过渡"""
    print_header(1, "movej — 关节空间运动", "余弦插值，零速启停")

    waypoints = [
        ([0.5, -1.0, 0.5, 1.0, -0.5, 0.5, 0.0], "前伸展开"),
        ([-0.5, -0.5, -0.3, 0.8, 0.5, -0.5, 0.5], "侧向扭转"),
        ([0.0, -1.5, 1.0, 1.5, -0.3, 0.8, -0.3], "高位折叠"),
        ([0.8, -0.3, 0.2, 0.3, -1.0, 0.3, 0.8], "后仰姿态"),
    ]

    for joints, name in waypoints:
        print(f"  → {name}: {[f'{j:.1f}' for j in joints]}")
        state = arm.movej(joints, duration=2.0)
        err = max(abs(state.joints[i] - joints[i]) for i in range(7))
        print(f"    误差: {err:.4f} rad")
        wait_viewer(viewer, 0.3)

    go_home(arm)
    print("  ✓ 完成")


# ─── Demo 2: movel — 笛卡尔直线 ──────────────────────────────────────────────

def demo_movel(arm, viewer, ik):
    """末端沿直线运动（笛卡尔空间）"""
    print_header(2, "movel — 笛卡尔直线运动", "末端直线插值 + IK 逆解")

    # 起始构型：距基座 0.55m，各方向 10cm 均可达
    start_q = [0.5, -1.2, 0.0, 1.5, 0.0, 0.8, 0.0]
    arm.movej(start_q, duration=2.0)

    ik.q = np.array(arm.state.joints)
    pos_start, _ = ik.get_ee_pose(np.array(arm.state.joints))
    print(f"  起始 EE 位置: [{pos_start[0]:.3f}, {pos_start[1]:.3f}, {pos_start[2]:.3f}]")

    # 矩形路径：每段 10cm
    targets = [
        (pos_start + np.array([0.10, 0.0, 0.0]), "X+ 前移 10cm"),
        (pos_start + np.array([0.10, 0.0, -0.10]), "Z- 下移 10cm"),
        (pos_start + np.array([0.0, 0.0, -0.10]), "X- 后移 10cm"),
        (pos_start.copy(), "Z+ 回到原点"),
    ]

    dt = 0.005
    for target_pos, name in targets:
        print(f"  → 直线运动: {name}")
        cur_q = np.array(arm.state.joints)
        cur_pos, _ = ik.get_ee_pose(cur_q)
        dist = np.linalg.norm(target_pos - cur_pos)
        duration = max(1.5, dist / 0.08)  # ~8cm/s
        steps = int(duration / dt)

        for step in range(steps):
            t = (step + 1) / steps
            alpha = 0.5 * (1.0 - math.cos(math.pi * t))
            interp_pos = cur_pos + alpha * (target_pos - cur_pos)
            q_now = np.array(arm.state.joints)
            q = ik.solve_ik(interp_pos, target_rot=None, q_init=q_now, max_iter=50)
            arm.servoj(q.tolist())
            if step % 10 == 0:
                viewer.sync()

        viewer.sync()
        final_pos, _ = ik.get_ee_pose(np.array(arm.state.joints))
        err = np.linalg.norm(final_pos - target_pos)
        print(f"    位置误差: {err*1000:.1f} mm")
        wait_viewer(viewer, 0.2)

    go_home(arm)
    print("  ✓ 完成")


# ─── Demo 3: movec — 圆弧运动 ────────────────────────────────────────────────

def demo_movec(arm, viewer, ik):
    """末端画圆弧"""
    print_header(3, "movec — 圆弧运动", "末端在空间中画圆")

    # 起始构型：工作空间中心，各方向有余量
    start_q = [0.5, -1.2, 0.0, 1.5, 0.0, 0.8, 0.0]
    arm.movej(start_q, duration=2.0)

    ik.q = np.array(arm.state.joints)
    pos_center, _ = ik.get_ee_pose(np.array(arm.state.joints))
    radius = 0.05  # 5cm 半径

    # 先移到圆上起始点（避免启动突变）
    first_point = pos_center + np.array([0.0, radius, 0.0])
    q_now = np.array(arm.state.joints)
    q_first = ik.solve_ik(first_point, target_rot=None, q_init=q_now, max_iter=100)
    for _ in range(100):
        arm.servoj(q_first.tolist())
    print(f"  圆心: [{pos_center[0]:.3f}, {pos_center[1]:.3f}, {pos_center[2]:.3f}]")
    print(f"  半径: {radius*100:.0f} cm")
    print(f"  画 2 圈...")

    dt = 0.005
    duration = 8.0
    steps = int(duration / dt)
    errors = []

    for step in range(steps):
        t = (step + 1) / steps
        angle = t * 2 * math.pi * 2  # 2 圈
        # 在 YZ 平面画圆
        offset = np.array([0.0, radius * math.cos(angle), radius * math.sin(angle)])
        target_pos = pos_center + offset
        q_now = np.array(arm.state.joints)
        q = ik.solve_ik(target_pos, target_rot=None, q_init=q_now, max_iter=50)
        arm.servoj(q.tolist())
        if step % 10 == 0:
            viewer.sync()
        # 每 100 步记录误差
        if step % 100 == 0:
            actual_pos, _ = ik.get_ee_pose(np.array(arm.state.joints))
            r_actual = np.linalg.norm(actual_pos - pos_center)
            errors.append(abs(r_actual - radius) * 1000)

    viewer.sync()
    if errors:
        print(f"  半径误差统计: 均值={np.mean(errors):.1f}mm, "
              f"最大={np.max(errors):.1f}mm, 最小={np.min(errors):.1f}mm")
    go_home(arm)
    print("  ✓ 完成")


# ─── Demo 4: servoj — 实时伺服 ────────────────────────────────────────────────

def demo_servoj(arm, viewer):
    """多关节正弦波组合 — 舞动效果"""
    print_header(4, "servoj — 实时伺服控制", "200Hz 多关节正弦波组合")

    # 起始位远离关节限位（J1 上限仅 0.26rad，必须从负值开始）
    init_joints = [0.3, -1.0, 0.0, 0.5, 0.0, 0.0, 0.0]
    arm.movej(init_joints, duration=1.5)

    duration = 5.0
    dt = 0.005
    steps = int(duration / dt)
    print(f"  大幅正弦波组合运动 {duration:.0f} 秒...")
    print(f"  J1: ±0.5 rad @ 0.4Hz | J2: ±0.4 rad @ 0.5Hz")
    print(f"  J3: ±0.6 rad @ 0.3Hz | J4: ±0.5 rad @ 0.6Hz")
    print(f"  J5: ±0.5 rad @ 0.7Hz | J6: ±0.4 rad @ 0.5Hz")

    max_errors = [0.0] * 7
    for step in range(steps):
        t = step * dt
        joints = list(init_joints)
        joints[0] += 0.5 * math.sin(2 * math.pi * 0.4 * t)
        joints[1] += 0.4 * math.sin(2 * math.pi * 0.5 * t)
        joints[2] += 0.6 * math.sin(2 * math.pi * 0.3 * t + 0.5)
        joints[3] += 0.5 * math.sin(2 * math.pi * 0.6 * t + 1.0)
        joints[4] += 0.5 * math.sin(2 * math.pi * 0.7 * t + 1.5)
        joints[5] += 0.4 * math.sin(2 * math.pi * 0.5 * t + 2.0)
        joints[6] += 0.5 * math.sin(2 * math.pi * 0.3 * t + 0.8)
        state = arm.servoj(joints)
        if t > 0.5:
            for i in range(7):
                err = abs(state.joints[i] - joints[i])
                if err > max_errors[i]:
                    max_errors[i] = err
        if step % 10 == 0:
            viewer.sync()

    viewer.sync()
    print(f"  稳态跟踪误差(max): {[f'{e:.4f}' for e in max_errors]} rad")
    print(f"  最大单关节误差: {max(max_errors):.4f} rad ({math.degrees(max(max_errors)):.1f}°)")
    go_home(arm)
    print("  ✓ 完成")


# ─── Demo 5: speedj — 关节速度控制 ────────────────────────────────────────────

def demo_speedj(arm, viewer):
    """关节速度控制 — 梯形加减速"""
    print_header(5, "speedj — 关节速度控制", "梯形速度曲线，加速→匀速→减速")

    motions = [
        ([0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0], 2.0, "J2 正转 0.8 rad/s"),
        ([0.0, -0.8, 0.0, 0.0, 0.0, 0.0, 0.0], 2.0, "J2 反转 0.8 rad/s"),
        ([0.3, 0.3, -0.3, 0.3, -0.3, 0.3, -0.3], 2.5, "全关节交替"),
    ]

    q_before = list(arm.state.joints)
    for vel, dur, name in motions:
        print(f"  → {name} ({dur}s)")
        # speedj 是内部阻塞调用，需要逐帧可视化
        dt = 0.005
        steps = int(dur / dt)
        acc = 3.0
        cur_vel = [0.0] * 7
        for step in range(steps):
            t = step * dt
            # 梯形加速
            for i in range(7):
                if abs(cur_vel[i]) < abs(vel[i]):
                    cur_vel[i] += math.copysign(acc * dt, vel[i])
                    if abs(cur_vel[i]) > abs(vel[i]):
                        cur_vel[i] = vel[i]
            target = [arm.state.joints[i] + cur_vel[i] * dt for i in range(7)]
            arm.servoj(target)
            if step % 10 == 0:
                viewer.sync()
        viewer.sync()
        displacement = [arm.state.joints[i] - q_before[i] for i in range(7)]
        active = [(i, d) for i, d in enumerate(displacement) if abs(d) > 0.01]
        for i, d in active:
            print(f"    J{i+1} 位移: {d:+.3f} rad")
        q_before = list(arm.state.joints)
        wait_viewer(viewer, 0.3)

    go_home(arm)
    print("  ✓ 完成")


# ─── Demo 6: speedl — 笛卡尔速度控制 ─────────────────────────────────────────

def demo_speedl(arm, viewer, ik):
    """笛卡尔速度控制 — 末端匀速平移"""
    print_header(6, "speedl — 笛卡尔速度控制", "末端匀速运动，IK 实时逆解")

    # 起始构型：工作空间中心
    start_q = [0.5, -1.2, 0.0, 1.5, 0.0, 0.8, 0.0]
    arm.movej(start_q, duration=2.0)

    ik.q = np.array(arm.state.joints)
    pos_start, _ = ik.get_ee_pose(np.array(arm.state.joints))
    print(f"  起始 EE 位置: [{pos_start[0]:.3f}, {pos_start[1]:.3f}, {pos_start[2]:.3f}]")

    # 矩形路径：5cm/s × 1.5s = 7.5cm 每段
    motions = [
        ([0.0, 0.0, -0.05], 1.5, "Z- 下移 5cm/s"),
        ([0.05, 0.0, 0.0], 1.5, "X+ 前移 5cm/s"),
        ([0.0, 0.0, 0.05], 1.5, "Z+ 上移 5cm/s"),
        ([-0.05, 0.0, 0.0], 1.5, "X- 后移 5cm/s"),
    ]

    dt = 0.005
    cumulative_offset = np.zeros(3)
    for vel, dur, name in motions:
        print(f"  → {name}")
        vel_arr = np.array(vel)
        steps = int(dur / dt)
        seg_start_pos, _ = ik.get_ee_pose(np.array(arm.state.joints))
        for step in range(steps):
            target_pos = seg_start_pos + vel_arr * (step + 1) * dt
            q_now = np.array(arm.state.joints)
            q = ik.solve_ik(target_pos, target_rot=None, q_init=q_now, max_iter=50)
            arm.servoj(q.tolist())
            if step % 10 == 0:
                viewer.sync()
        viewer.sync()
        cumulative_offset += vel_arr * dur
        pos_now, _ = ik.get_ee_pose(np.array(arm.state.joints))
        expected_pos = pos_start + cumulative_offset
        err = np.linalg.norm(pos_now - expected_pos) * 1000
        print(f"    EE: [{pos_now[0]:.3f}, {pos_now[1]:.3f}, {pos_now[2]:.3f}], 累积误差: {err:.1f}mm")
        wait_viewer(viewer, 0.2)

    go_home(arm)
    print("  ✓ 完成")


# ─── Demo 7: 轨迹融合 ────────────────────────────────────────────────────────

def demo_blending(arm, viewer):
    """多 waypoint 轨迹平滑融合"""
    print_header(7, "Trajectory Blending — 多段轨迹融合",
                 "五次多项式融合，过渡无速度跳变")

    planner = TrajectoryPlanner(num_joints=7, control_freq=200.0)
    waypoints_data = [
        [0.5, -0.8, 0.5, 0.8, -0.3, 0.4, 0.2],
        [-0.5, -1.2, 0.8, 1.2, 0.3, -0.3, -0.3],
        [0.3, -0.5, -0.2, 0.5, -0.8, 0.6, 0.5],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    planner.add_waypoint(Waypoint(joints=waypoints_data[0], duration=1.5, blend=0.3))
    planner.add_waypoint(Waypoint(joints=waypoints_data[1], duration=1.5, blend=0.3))
    planner.add_waypoint(Waypoint(joints=waypoints_data[2], duration=1.5, blend=0.3))
    planner.add_waypoint(Waypoint(joints=waypoints_data[3], duration=1.5))

    frames = planner.compute(arm.state.joints)
    print(f"  4 个 waypoint, blend=0.3s")
    print(f"  总帧数: {len(frames)}")
    print(f"  执行中...")

    # 逐帧执行并可视化
    max_err = 0.0
    for i, frame in enumerate(frames):
        arm.servoj(frame.joints)
        if i % 10 == 0:
            viewer.sync()
    viewer.sync()

    # 最终误差（应回到零位）
    final_err = max(abs(arm.state.joints[j]) for j in range(7))
    print(f"  最终回零误差: {final_err:.4f} rad")
    print("  ✓ 完成")


# ─── Demo 8: 录制与回放 ───────────────────────────────────────────────────────

def demo_record_replay(arm, viewer):
    """录制一段运动并回放"""
    print_header(8, "Record + Replay — 轨迹录制与回放",
                 "录制运动 → 正常回放 → 2x 加速回放")

    # 录制一段运动（逐帧可视化）
    print("  录制中...")
    arm.start_recording()
    arm.movej([0.5, -0.8, 0.5, 0.8, -0.3, 0.4, 0.2], duration=1.5)
    arm.movej([-0.3, -1.0, 0.3, 1.0, 0.5, -0.4, -0.2], duration=1.5)
    arm.movej([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], duration=1.5)
    recording = arm.stop_recording()
    print(f"  录制完成: {recording.num_frames} 帧, {recording.duration:.1f} 秒")

    # 正常速度回放
    print("  1x 速度回放...")
    arm.replay(recording, speed_factor=1.0)
    viewer.sync()
    wait_viewer(viewer, 0.5)

    # 2x 加速回放
    print("  2x 加速回放...")
    arm.replay(recording, speed_factor=2.0)
    viewer.sync()

    print("  ✓ 完成")


# ─── Demo 9: Freedrive 模拟 ───────────────────────────────────────────────────

def demo_freedrive(arm, viewer, backend):
    """零重力示教模式 — 纯重力补偿，手臂可自由运动"""
    print_header(9, "Freedrive — 零重力示教模式",
                 "kp=0 + 重力补偿，施加外力后手臂漂移")

    # 移到一个有重力影响的姿态
    start_q = [0.0, -0.8, 0.5, 0.8, 0.0, 0.3, 0.0]
    arm.movej(start_q, duration=1.5)

    print("  启动 freedrive...")
    arm.freedrive_enable(damping=0.3)
    
    wait_viewer(viewer, 1.0)

    # 模拟外力推动（通过直接操作仿真器的外力）
    print("  施加模拟外力 (2s)...")
    t0 = time.monotonic()
    while time.monotonic() - t0 < 2.0:
        # 在关节 2 和 4 施加脉冲力矩
        t = time.monotonic() - t0
        backend._data.qfrc_applied[1] += 1.5 * math.sin(2 * math.pi * 0.5 * t)
        backend._data.qfrc_applied[3] += 1.0 * math.sin(2 * math.pi * 0.3 * t)
        viewer.sync()
        time.sleep(0.016)

    print("  外力撤除，自由阻尼衰减 (2s)...")
    wait_viewer(viewer, 2.0)

    arm.freedrive_disable()
    
    print(f"  最终位置: [{', '.join(f'{j:.2f}' for j in arm.state.joints)}]")

    go_home(arm)
    print("  ✓ 完成")


# ─── Demo 10: Force Mode ─────────────────────────────────────────────────────

def demo_force_mode(arm, viewer):
    """力/位混合控制 — Z轴力控，XY位控"""
    print_header(10, "Force Mode — 力/位混合控制",
                 "Z 轴施加恒力, XY 保持位置")

    # 先到起始位
    start_q = [0.0, -0.8, 0.4, 0.8, 0.0, 0.3, 0.0]
    arm.movej(start_q, duration=1.5)

    print("  Z 轴施力 -5N (下压), XY 保持位置")
    print("  运行 3 秒...")

    dt = 0.005
    steps = int(3.0 / dt)
    for step in range(steps):
        arm.force_mode(
            selection=[0, 0, 1, 0, 0, 0],
            wrench=[0, 0, -5.0, 0, 0, 0],
            limits=[0, 0, 0.05, 0, 0, 0],
        )
        if step % 10 == 0:
            viewer.sync()

    arm.force_mode_disable()
    
    print(f"  末端下移后位置: [{', '.join(f'{j:.2f}' for j in arm.state.joints)}]")

    # 换方向：Y轴力控
    print("  Y 轴施力 +3N (侧推), XZ 保持位置")
    print("  运行 2 秒...")

    steps = int(2.0 / dt)
    for step in range(steps):
        arm.force_mode(
            selection=[0, 1, 0, 0, 0, 0],
            wrench=[0, 3.0, 0, 0, 0, 0],
            limits=[0, 0.05, 0, 0, 0, 0],
        )
        if step % 20 == 0:
            viewer.sync()

    arm.force_mode_disable()
    

    go_home(arm)
    print("  ✓ 完成")


# ─── Demo 11: 刚度预设 ───────────────────────────────────────────────────────

def demo_stiffness(arm, viewer, backend):
    """刚度预设对比 — stiff / medium / soft"""
    print_header(11, "Stiffness — 刚度预设切换",
                 "同一目标, 不同阻抗响应对比")

    target = [0.0, -1.2, 0.8, 1.0, -0.5, 0.6, 0.0]

    results = {}
    for preset in ["stiff", "medium", "soft"]:
        backend.reset()
        arm.enable()
        arm.set_stiffness(preset)
        viewer.sync()

        state = arm.movej(target, duration=1.5)

        err = max(abs(state.joints[i] - target[i]) for i in range(7))
        results[preset] = err
        omega = {'stiff': 40, 'medium': 20, 'soft': 8}[preset]
        print(f"  [{preset:6s}] ωn={omega:2d} → 跟踪误差: {err:.4f} rad")
        wait_viewer(viewer, 0.8)

    # 展示 soft 模式的柔顺效果
    print("\n  SOFT 模式下快速运动的柔顺效果:")
    backend.reset()
    arm.enable()
    arm.set_stiffness("soft")
    viewer.sync()
    state = arm.movej(target, duration=0.8)
    wait_viewer(viewer, 1.5)
    err = max(abs(state.joints[i] - target[i]) for i in range(7))
    print(f"    快速 soft: 误差 {err:.4f} rad (可见明显延迟)")

    # 恢复
    arm.set_stiffness("stiff")
    backend.reset()
    arm.enable()
    viewer.sync()
    print("  ✓ 完成")


# ─── Demo 12: 外力估计 ───────────────────────────────────────────────────────

def demo_wrench_estimation(arm, viewer, backend):
    """外力估计 — 通过力矩残差推算末端外力"""
    print_header(12, "External Wrench — 外力估计",
                 "基于力矩残差 + Jacobian 转置逆解")

    start_q = [0.0, -0.8, 0.4, 0.8, 0.0, 0.3, 0.0]
    arm.movej(start_q, duration=1.5)

    # 无外力基线
    wrench_base = arm.get_external_wrench()
    print(f"  基线力: [{', '.join(f'{w:.2f}' for w in wrench_base[:3])}] N")
    print(f"  基线矩: [{', '.join(f'{w:.2f}' for w in wrench_base[3:])}] Nm")

    # 施加模拟外力
    print("\n  施加模拟外力...")
    backend._data.qfrc_applied[1] += 3.0
    backend._data.qfrc_applied[3] += 2.0
    # 跑几步让状态更新
    for _ in range(20):
        arm.servoj(list(arm.state.joints))
    

    wrench_ext = arm.get_external_wrench()
    print(f"  受力后:")
    print(f"    力: [{', '.join(f'{w:.2f}' for w in wrench_ext[:3])}] N")
    print(f"    矩: [{', '.join(f'{w:.2f}' for w in wrench_ext[3:])}] Nm")
    diff = np.linalg.norm(wrench_ext[:3] - wrench_base[:3])
    print(f"    力变化量: {diff:.2f} N")

    go_home(arm)
    print("  ✓ 完成")


# ─── Demo 13: 碰撞检测 ───────────────────────────────────────────────────────

def demo_collision(arm, viewer, backend):
    """碰撞检测 — 力矩残差超阈值触发"""
    print_header(13, "Collision Detection — 碰撞检测",
                 "力矩残差超阈值即判定碰撞")

    start_q = [0.0, -0.6, 0.3, 0.6, 0.0, 0.3, 0.0]
    arm.movej(start_q, duration=1.5)

    # 正常状态检查
    collision = arm.check_collision()
    print(f"  正常状态: collision={collision}")

    # 模拟碰撞（施加大力矩）
    print("  模拟碰撞冲击...")
    backend._data.qfrc_applied[2] += 8.0
    backend._data.qfrc_applied[4] += 6.0
    for _ in range(10):
        arm.servoj(list(arm.state.joints))
    

    collision = arm.check_collision()
    print(f"  冲击后: collision={collision}")

    # 清除外力
    backend._data.qfrc_applied[:] = 0.0
    for _ in range(20):
        arm.servoj(list(arm.state.joints))
    

    go_home(arm)
    print("  ✓ 完成")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  onero SDK — MuJoCo 可视化全功能演示")
    print("  A1 7-DOF 机械臂")
    print("=" * 60)
    print("\n启动 MuJoCo 渲染器...")

    backend, viewer = make_viewer_backend("A1")
    arm = Arm("A1", backend=backend)
    arm.enable()
    viewer.sync()

    # 初始化 IK solver (pinocchio DLS, 不需要 pink)
    urdf_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "onero_sdk", "urdf", "A1", "a1_r.urdf"
    )
    ik = IKSolver(urdf_path)

    print("渲染器已启动。\n")
    wait_viewer(viewer, 1.0)

    demos = [
        ("movej", lambda: demo_movej(arm, viewer)),
        ("movel", lambda: demo_movel(arm, viewer, ik)),
        ("movec", lambda: demo_movec(arm, viewer, ik)),
        ("servoj", lambda: demo_servoj(arm, viewer)),
        ("speedj", lambda: demo_speedj(arm, viewer)),
        ("speedl", lambda: demo_speedl(arm, viewer, ik)),
        ("blending", lambda: demo_blending(arm, viewer)),
        ("record+replay", lambda: demo_record_replay(arm, viewer)),
        ("freedrive", lambda: demo_freedrive(arm, viewer, backend)),
        ("force_mode", lambda: demo_force_mode(arm, viewer)),
        ("stiffness", lambda: demo_stiffness(arm, viewer, backend)),
        ("wrench", lambda: demo_wrench_estimation(arm, viewer, backend)),
        ("collision", lambda: demo_collision(arm, viewer, backend)),
    ]

    try:
        for name, fn in demos:
            fn()
            wait_viewer(viewer, 1.0)

        print(f"\n{'='*60}")
        print("  所有 13 项演示完成！")
        print("  渲染器保持打开，关闭窗口或 Ctrl+C 退出。")
        print(f"{'='*60}\n")

        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\n  用户中断，退出。")
    finally:
        arm.disable()
        viewer.close()


if __name__ == "__main__":
    main()
