"""
坐标系与位姿 — 理解 FK/IK，关节空间 vs 笛卡尔空间。

关节空间 (Joint Space):
  - 7 个关节角度 [q1, q2, ..., q7] (单位: rad)
  - 直接控制每个电机的角度

笛卡尔空间 (Cartesian / Task Space):
  - 末端位置 [x, y, z] (单位: 米)
  - 末端姿态: 旋转矩阵 / 四元数 / 旋转向量

FK (正运动学): 关节角 → 末端位姿
IK (逆运动学): 末端位姿 → 关节角

运行:
    pip install onero-sdk[ik]
    python examples/10_coordinate_systems.py
"""

import numpy as np  # noqa: F401 (kept for user experimentation)
from onero_sdk import Arm

arm = Arm("A1", channel="can0")
arm.enable()

try:
    # === 1. 关节空间运动 ===
    print("1. 关节空间: 直接指定 7 个角度")
    arm.movej([0.0, -0.5, 0.3, -1.0, 0.0, 0.5, 0.0], duration=2.0)
    print(f"   当前关节: {[f'{j:.2f}' for j in arm.state.joints]}")

    # === 2. 正运动学: 关节角 → 末端位置 ===
    print("\n2. FK: 关节角 → 末端位置")
    pos, rot = arm.get_pose()
    print(f"   末端位置: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] m")
    print(f"   末端姿态(旋转矩阵 3x3):\n{rot}")

    # === 3. 笛卡尔直线运动 (movel) ===
    print("\n3. movel: 末端沿直线移动到目标位置")
    arm.movel(
        position=[0.3, 0.0, 0.25, 0.0, 0.0, 0.0],  # 目标 [x, y, z, rx, ry, rz]
        duration=2.0,
    )
    pos, _ = arm.get_pose()
    print(f"   到达: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] m")

    # === 4. 位姿的多种表示方式 ===
    print("\n4. 位姿表示 — movel 支持多种格式:")
    print("   (a) 位置 + 旋转矩阵:")
    print("       arm.movel(position=[x,y,z], rotation=R_3x3)")
    print("   (b) 位置 + 四元数:")
    print("       arm.movel(position=[x,y,z], rotation=[qx,qy,qz,qw])")
    print("   (c) 位置 + 旋转向量:")
    print("       arm.movel(position=[x,y,z], rotation=[rx,ry,rz])")
    print("   (d) 6D 位姿 (位置+旋转向量):")
    print("       arm.movel(position=[x,y,z,rx,ry,rz])")

    # === 5. movep: 末端到达目标（路径不限直线） ===
    print("\n5. movep: 到达目标位姿（路径不是严格直线，但更快）")
    arm.movep(position=[0.25, 0.1, 0.2, 0.0, 0.0, 0.0], duration=2.0)
    pos, _ = arm.get_pose()
    print(f"   到达: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] m")

finally:
    arm.disable()
