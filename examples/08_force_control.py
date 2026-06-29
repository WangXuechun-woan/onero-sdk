"""
力控 / 柔顺控制 — 接触面操作。

演示 force_mode: 在某些轴上施加力，其余轴保持位置。
典型应用: 打磨、擦拭、装配插入、按压接触等。

原理:
  force_mode 中指定的力轴用力控(恒力/阻抗)，
  其余轴保持当前位置(位置控制)。

注意:
  force_mode 是单帧指令(需要在循环中持续调用)，
  类似 servoj，而非像 movej 那样自带阻塞。

运行:
    python examples/08_force_control.py
"""

import time
from onero_sdk import Arm

arm = Arm("A1R", channel="can0")
arm.enable()

try:
    # 先移动到操作位姿
    arm.movej([0.0, -0.5, 0.3, -1.0, 0.0, 0.5, 0.0], duration=2.0)

    # === 场景 1: Z 轴恒力下压（打磨/擦拭） ===
    print("场景 1: Z 轴下压 5N (持续 3 秒)")
    # selection: [fx, fy, fz, tx, ty, tz], 1=力控, 0=位置控
    # wrench: [fx, fy, fz, tx, ty, tz] (N, Nm)
    t0 = time.time()
    while time.time() - t0 < 3.0:
        arm.force_mode(
            selection=[0, 0, 1, 0, 0, 0],    # Z 轴力控，其余位置控
            wrench=[0, 0, -5, 0, 0, 0],      # Z 方向施加 -5N
        )
    arm.force_mode_disable()
    print("  完成")

    # === 场景 2: 多轴力控（柔顺装配） ===
    print("\n场景 2: XYZ 平面自由 + Z 轴下压 (持续 3 秒)")
    t0 = time.time()
    while time.time() - t0 < 3.0:
        arm.force_mode(
            selection=[1, 1, 1, 0, 0, 0],    # XYZ 全部力控
            wrench=[0, 0, -3, 0, 0, 0],      # Z 方向施加 -3N
        )
    arm.force_mode_disable()
    print("  完成")

    # === 场景 3: 用 frame= 指定力控坐标系 ===
    print("\n场景 3: 在倾斜面上施加法向力")
    print("  方式 A: 用 frame= 指定力控参考坐标系")
    # 注册一个绕 Y 轴旋转 30° 的自定义坐标系
    import math
    import numpy as np
    from onero_sdk.geometry import rotvec_to_matrix
    R_tilted = rotvec_to_matrix([0, math.radians(30), 0])
    arm.register_frame("tilted_surface", rotation=R_tilted)

    t0 = time.time()
    while time.time() - t0 < 2.0:
        arm.force_mode(
            selection=[0, 0, 1, 0, 0, 0],
            wrench=[0, 0, -5, 0, 0, 0],
            frame="tilted_surface",  # 力方向沿倾斜面法线
        )
    arm.force_mode_disable()
    print("  完成")

    # 方式 B: 用 task_frame (旧 API，仍然兼容)
    print("  方式 B: 用 task_frame= (旧 API 兼容)")
    t0 = time.time()
    while time.time() - t0 < 2.0:
        arm.force_mode(
            selection=[0, 0, 1, 0, 0, 0],
            wrench=[0, 0, -5, 0, 0, 0],
            task_frame=[0, 0, 0, 0, math.radians(30), 0],  # 绕 Y 轴旋转 30°
        )
    arm.force_mode_disable()
    print("  完成")

    # 回到初始位置
    arm.movej([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], duration=2.0)

finally:
    arm.disable()
