"""
双臂坐标系示例 — 在不同坐标系下做协调运动。

DualArm 的核心价值: 两条臂共享统一的世界坐标系,
使得协调运动、视觉引导、工件坐标系编程都变得自然。

演示内容:
  1. world 坐标系: 两臂在同一世界坐标系中运动
  2. 单臂 frame="base": 回退到各自 base 坐标系
  3. tool 坐标系: 两臂各自沿工具方向进给
  4. 自定义坐标系: 相机坐标系 + 工件坐标系
  5. force_mode + frame: 在倾斜面上协调力控

运行:
    pip install onero-sdk[sim,dynamics]
    python examples/dual_arm.py
"""

import time
import math
import numpy as np
from onero_sdk import DualArm
from onero_sdk.geometry import rotvec_to_matrix

# ═══════════════════════════════════════════════════════════════
# 创建双臂 — arm_spacing 定义两臂间距
# left: A1L + side_mount_left, base_position=[0, +0.15, 0]
# right: A1R + side_mount_right, base_position=[0, -0.15, 0]
# ═══════════════════════════════════════════════════════════════
dual = DualArm.create(
    left_channel="can0",
    right_channel="can1",
    arm_spacing=0.30,  # 30cm between bases along Y
)

with dual:
    # ─── 1. world 坐标系: 对称运动 ─────────────────────────────
    print("1. frame='world' — 两臂在统一世界坐标系中运动")
    print("   左臂到 [0.3, 0.15, 0.4], 右臂到 [0.3, -0.15, 0.4]")
    dual.movel(
        left_position=[0.3, 0.15, 0.4, 0, 0, 0],
        right_position=[0.3, -0.15, 0.4, 0, 0, 0],
        duration=3.0,
        frame="world",
    )

    # 获取两臂末端在世界坐标系中的位姿
    (l_pos, l_rot), (r_pos, r_rot) = dual.get_poses(frame="world")
    print(f"   左臂位置(world): [{l_pos[0]:.3f}, {l_pos[1]:.3f}, {l_pos[2]:.3f}]")
    print(f"   右臂位置(world): [{r_pos[0]:.3f}, {r_pos[1]:.3f}, {r_pos[2]:.3f}]")

    # ─── 2. 单臂独立运动 (base / world) ───────────────────────
    print("\n2. 单臂独立运动 — 各自可以用不同 frame")
    # 左臂在世界坐标系中上升 5cm
    l_pos_world, l_rot_world = dual.left.get_pose(frame="world")
    target = l_pos_world.copy()
    target[2] += 0.05  # world Z 方向上升
    dual.left.movel(target, l_rot_world, duration=1.5, frame="world")
    print("   左臂: world Z +5cm 完成")

    # 右臂在 base 坐标系中运动
    r_pos_base, r_rot_base = dual.right.get_pose(frame="base")
    target_base = r_pos_base.copy()
    target_base[0] += 0.03  # base X 方向移动
    dual.right.movel(target_base, r_rot_base, duration=1.5, frame="base")
    print("   右臂: base X +3cm 完成")

    # ─── 3. tool 坐标系: 沿工具方向进给 ──────────────────────
    print("\n3. frame='tool' — 沿各自工具方向进给")
    # 给两臂设置 TCP
    dual.left.set_tcp([0, 0, 0.12])   # 左臂工具长 12cm
    dual.right.set_tcp([0, 0, 0.12])  # 右臂工具长 12cm
    print("   两臂 TCP 设置: [0, 0, 0.12]")

    # 两臂各自沿工具 Z 方向前进 3cm
    dual.left.movel([0, 0, 0.03, 0, 0, 0], frame="tool", duration=1.0)
    dual.right.movel([0, 0, 0.03, 0, 0, 0], frame="tool", duration=1.0)
    print("   两臂各沿 tool Z 前进 3cm 完成")

    # 速度控制: 沿工具 Z 轴以恒定速度进给
    dual.left.speedl([0, 0, 0.03, 0, 0, 0], duration=1.0, frame="tool")
    print("   左臂: tool Z 方向 3cm/s 进给 1s 完成")

    dual.left.clear_tcp()
    dual.right.clear_tcp()

    # ─── 4. 自定义坐标系: 相机坐标系 ─────────────────────────
    print("\n4. 自定义坐标系 — 视觉引导协调抓取")
    # 假设顶部相机通过手眼标定得到:
    R_cam = np.array([
        [0, -1,  0],
        [1,  0,  0],
        [0,  0,  1],
    ], dtype=float)
    p_cam = [0.3, 0.0, 1.0]  # 相机在世界系中的位置

    # 两臂共享同一个相机坐标系
    dual.left.register_frame("camera", rotation=R_cam, position=p_cam)
    dual.right.register_frame("camera", rotation=R_cam, position=p_cam)
    print(f"   两臂注册 'camera' 坐标系: origin={p_cam}")

    # 视觉检测到物体在相机系中的两个抓取点
    # 左臂去抓左侧, 右臂去抓右侧 (在相机坐标系中表达)
    left_grasp_cam = dual.left.get_pose(frame="camera")
    right_grasp_cam = dual.right.get_pose(frame="camera")
    print(f"   左臂位置(camera): [{left_grasp_cam[0][0]:.3f}, "
          f"{left_grasp_cam[0][1]:.3f}, {left_grasp_cam[0][2]:.3f}]")
    print(f"   右臂位置(camera): [{right_grasp_cam[0][0]:.3f}, "
          f"{right_grasp_cam[0][1]:.3f}, {right_grasp_cam[0][2]:.3f}]")
    # 实际使用: dual.left.movel(target_in_cam, frame="camera")

    dual.left.unregister_frame("camera")
    dual.right.unregister_frame("camera")

    # ─── 5. 工件坐标系: 协调装配 ─────────────────────────────
    print("\n5. 工件坐标系 — 在工件局部坐标系中编程")
    # 3 点标定工件坐标系（在世界坐标系下测量的 3 个点）
    O  = np.array([0.35, 0.0, 0.2])
    Px = np.array([0.45, 0.0, 0.2])
    Py = np.array([0.35, 0.1, 0.2])

    x_axis = (Px - O) / np.linalg.norm(Px - O)
    y_raw = Py - O
    z_axis = np.cross(x_axis, y_raw)
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    R_wp = np.column_stack([x_axis, y_axis, z_axis])

    dual.left.register_frame("workpiece", rotation=R_wp, position=O)
    dual.right.register_frame("workpiece", rotation=R_wp, position=O)
    print(f"   注册 'workpiece' 坐标系: origin={O.tolist()}")

    # 左臂到工件坐标系中的 [0, 0.05, 0.1] (工件左上方)
    # 右臂到工件坐标系中的 [0, -0.05, 0.1] (工件右上方)
    # dual.left.movel([0, 0.05, 0.1, 0, 0, 0], frame="workpiece", duration=2.0)
    # dual.right.movel([0, -0.05, 0.1, 0, 0, 0], frame="workpiece", duration=2.0)
    print("   可用: arm.movel([x,y,z,...], frame='workpiece')")

    dual.left.unregister_frame("workpiece")
    dual.right.unregister_frame("workpiece")

    # ─── 6. force_mode + frame: 协调力控 ─────────────────────
    print("\n6. force_mode + frame — 双臂协调力控")
    # 右臂在世界 Z 方向施加下压力
    print("   右臂: world Z 方向施力 -5N (1 秒)")
    t0 = time.time()
    while time.time() - t0 < 1.0:
        dual.right.force_mode(
            selection=[0, 0, 1, 0, 0, 0],
            wrench=[0, 0, -5, 0, 0, 0],
            frame="world",
        )
    dual.right.force_mode_disable()
    print("   完成")

    # ─── 回零 ────────────────────────────────────────────────
    dual.home(duration=3.0)
    print("\n双臂已回零。")
