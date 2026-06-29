"""
坐标系 (frame=) — 在不同坐标系下做定位移动。

所有笛卡尔 API（movel / movep / servop / speedl / force_mode / get_pose）
均支持 frame= 参数，可指定运动的参考坐标系。

内置坐标系:
  "base"   — 机械臂 base_link 原点，Z 沿第一关节轴
  "world"  — 世界坐标系（X=前, Y=左, Z=上），由 mounting + base_position 确定
  "tool"   — 工具坐标系（相对增量运动，UR 风格）

自定义坐标系:
  通过 register_frame(name, rotation, position) 注册
  典型用例: 相机坐标系、工件坐标系、工装坐标系

运行:
    pip install onero-sdk[ik]
    python examples/10_coordinate_systems.py
"""

import math
import numpy as np
from onero_sdk import Arm
from onero_sdk.geometry import rotvec_to_matrix

arm = Arm("A1R", channel="can0", mounting="side_mount_right", world_frame=True)
arm.enable()

try:
    # 先到一个安全的初始位姿
    arm.movej([0.0, -0.5, 0.3, -1.0, 0.0, 0.5, 0.0], duration=2.0)

    # ═══════════════════════════════════════════════
    # 1. base 坐标系
    # ═══════════════════════════════════════════════
    print("1. frame='base' — 机械臂基座坐标系")
    pos_base, rot_base = arm.get_pose(frame="base")
    print(f"   当前位置(base): [{pos_base[0]:.3f}, {pos_base[1]:.3f}, {pos_base[2]:.3f}]")
    # 在 base 坐标系中做小位移
    target_base = pos_base.copy()
    target_base[2] += 0.05  # base Z 方向抬高 5cm
    arm.movel(target_base, rot_base, duration=1.5, frame="base")
    print("   → base Z 方向抬高 5cm 完成")

    # ═══════════════════════════════════════════════
    # 2. world 坐标系
    # ═══════════════════════════════════════════════
    print("\n2. frame='world' — 世界坐标系 (X=前, Y=左, Z=上)")
    pos_world, rot_world = arm.get_pose(frame="world")
    print(f"   当前位置(world): [{pos_world[0]:.3f}, {pos_world[1]:.3f}, {pos_world[2]:.3f}]")
    # 在世界坐标系中做位移: 向前(X)移动 5cm
    target_world = pos_world.copy()
    target_world[0] += 0.05
    arm.movel(target_world, rot_world, duration=1.5, frame="world")
    print("   → world X 方向前进 5cm 完成")

    # ═══════════════════════════════════════════════
    # 3. tool 坐标系（相对增量运动）
    # ═══════════════════════════════════════════════
    print("\n3. frame='tool' — 工具坐标系（相对当前 TCP 位姿的增量）")
    # 必须先设置 TCP
    arm.set_tcp([0, 0, 0.10])  # 假设工具长 10cm
    print("   TCP 设置: [0, 0, 0.10]")

    # tool frame 中的值是增量，不是绝对目标
    # [0, 0, 0.03] = 沿工具 Z 轴前进 3cm
    arm.movel([0, 0, 0.03, 0, 0, 0], frame="tool", duration=1.0)
    print("   → 沿工具 Z 轴前进 3cm 完成")

    # 沿工具 X 轴横移
    arm.movel([0.02, 0, 0, 0, 0, 0], frame="tool", duration=1.0)
    print("   → 沿工具 X 轴横移 2cm 完成")

    arm.clear_tcp()

    # ═══════════════════════════════════════════════
    # 4. 自定义坐标系: 相机坐标系
    # ═══════════════════════════════════════════════
    print("\n4. 自定义坐标系 — 相机坐标系 (camera)")
    # 假设相机固定在工作台上方，通过手眼标定获得:
    # - R_cam_to_world: 相机坐标系旋转到世界坐标系的旋转矩阵
    # - p_cam_in_world: 相机原点在世界坐标系中的位置
    R_cam_to_world = np.array([
        [0, -1,  0],
        [1,  0,  0],
        [0,  0,  1],
    ], dtype=float)  # 相机 X → 世界 -Y, 相机 Y → 世界 X
    p_cam_in_world = [0.5, 0.0, 1.0]

    arm.register_frame("camera", rotation=R_cam_to_world, position=p_cam_in_world)
    print(f"   注册 'camera' 坐标系: origin={p_cam_in_world}")

    # 获取末端在相机坐标系中的位置
    pos_cam, _ = arm.get_pose(frame="camera")
    print(f"   当前位置(camera): [{pos_cam[0]:.3f}, {pos_cam[1]:.3f}, {pos_cam[2]:.3f}]")

    # 直接用视觉检测结果下发运动指令（无需手动坐标转换）
    # 假设视觉检测到目标在相机系 [0.1, -0.05, 0.6]
    # arm.movel([0.1, -0.05, 0.6, 0, 0, 0], frame="camera", duration=2.0)

    # ═══════════════════════════════════════════════
    # 5. 自定义坐标系: 工件坐标系 (3 点标定)
    # ═══════════════════════════════════════════════
    print("\n5. 自定义坐标系 — 工件坐标系 (3 点标定)")
    # 示教 3 个点（在世界坐标系下）来定义工件坐标系
    O  = np.array([0.4, 0.1, 0.05])   # 工件原点
    Px = np.array([0.5, 0.1, 0.05])   # X 方向上的一点
    Py = np.array([0.4, 0.2, 0.05])   # XY 平面内的一点

    x_axis = (Px - O) / np.linalg.norm(Px - O)
    y_raw = Py - O
    z_axis = np.cross(x_axis, y_raw)
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    R_workpiece = np.column_stack([x_axis, y_axis, z_axis])

    arm.register_frame("workpiece", rotation=R_workpiece, position=O)
    print(f"   注册 'workpiece' 坐标系: origin={O.tolist()}")

    # 在工件坐标系中编程（所有坐标都相对工件原点）
    # arm.movel([0.05, 0.02, 0.01, 0, 0, 0], frame="workpiece", duration=1.5)
    print("   在工件系中 [0.05, 0.02, 0.01] 就是工件表面的一个点")

    # ═══════════════════════════════════════════════
    # 6. 速度控制 + 坐标系
    # ═══════════════════════════════════════════════
    print("\n6. speedl + frame — 在指定坐标系中做速度控制")
    arm.set_tcp([0, 0, 0.10])
    # 沿工具 Z 轴以 5cm/s 进给 1 秒
    arm.speedl([0, 0, 0.05, 0, 0, 0], duration=1.0, frame="tool")
    print("   → 沿 tool Z 以 5cm/s 进给 1s 完成")
    arm.clear_tcp()

    # ═══════════════════════════════════════════════
    # 7. 力控 + 坐标系
    # ═══════════════════════════════════════════════
    print("\n7. force_mode + frame — 在指定坐标系中施加力")
    import time
    # 注册一个倾斜 30° 的表面坐标系
    R_tilted = rotvec_to_matrix([0, math.radians(30), 0])
    arm.register_frame("tilted_surface", rotation=R_tilted)
    # 沿倾斜面法线方向（该坐标系的 Z 轴）施加力
    t0 = time.time()
    while time.time() - t0 < 1.0:
        arm.force_mode(
            selection=[0, 0, 1, 0, 0, 0],
            wrench=[0, 0, -5, 0, 0, 0],
            frame="tilted_surface",
        )
    arm.force_mode_disable()
    print("   → 沿 tilted_surface Z 轴施加 5N 完成")

    # ═══════════════════════════════════════════════
    # 清理
    # ═══════════════════════════════════════════════
    arm.unregister_frame("camera")
    arm.unregister_frame("workpiece")
    arm.unregister_frame("tilted_surface")
    print("\n清理完成。")

finally:
    arm.disable()
