"""
工具坐标系 (TCP) — 夹爪/灵巧手的坐标变换。

当末端安装了夹爪、灵巧手或其他工具时，需要设置 TCP 偏移，
使所有运动指令以工具尖端(而非法兰面)为参考点。

概念:
  法兰面 (flange):  机械臂最后一个关节输出的参考面
  TCP (Tool Center Point): 工具的实际作用点
  TCP 偏移:  从法兰面到 TCP 的变换 [x, y, z, rx, ry, rz]

    T_base_tcp = T_base_flange × T_flange_tcp
                 (FK 输出)       (用户设置)

示例场景:
  - 夹爪尖端离法兰面 15cm → tcp_offset = [0, 0, 0.15, 0, 0, 0]
  - 工具安装有角度 → tcp_offset = [0, 0, 0.1, 0, 0.3, 0]

运行:
    python examples/13_tool_frame.py
"""

import math
import numpy as np
from onero_sdk import Arm

arm = Arm("A1R", channel="can0")
arm.enable()

try:
    # === 1. 设置 TCP 偏移 ===
    print("1. 设置 TCP (工具坐标系偏移)")
    print()
    print("   假设末端安装了一个 15cm 的平行夹爪:")
    print("   夹爪尖端在法兰面 Z 轴正方向 0.15m 处")
    print()

    # TCP 偏移:
    # position: [x, y, z] 法兰面坐标系下的平移 (m)
    # rotation: 旋转向量 [rx, ry, rz] (rad), 可选
    arm.set_tcp(position=[0.0, 0.0, 0.15])
    print(f"   TCP 设置完成: position=[0, 0, 0.15]")
    print(f"   现在 movel 的目标位置是夹爪尖端的位置，不是法兰面!")

    # === 2. 使用 TCP 后的运动 ===
    print("\n2. 使用 TCP 运动")
    print("   让夹爪尖端到达 [0.30, 0.0, 0.10]:")
    arm.movel(position=[0.30, 0.0, 0.10, 0.0, 0.0, 0.0], duration=2.0)
    pos, rot = arm.get_pose()
    print(f"   末端位姿 (TCP): [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    # 注: 法兰面实际在 [0.30, 0.0, 0.25] (高出 15cm)

    # === 3. 不同工具的 TCP 设置举例 ===
    print("\n3. 常见工具的 TCP 偏移示例:")
    print()
    print("   (a) 平行夹爪 (直装, 长 15cm):")
    print("       arm.set_tcp(position=[0, 0, 0.15])")
    print()
    print("   (b) 灵巧手 (长 20cm, 指尖作为 TCP):")
    print("       arm.set_tcp(position=[0, 0, 0.20])")
    print()
    print("   (c) 吸盘 (45° 安装角, 长 10cm):")
    print("       arm.set_tcp(position=[0, 0, 0.10], rotation=[0, 0.785, 0])")
    print()
    print("   (d) 焊枪 (偏心安装 + 倾斜):")
    print("       arm.set_tcp(position=[0.05, 0, 0.12], rotation=[0, 0.52, 0])")

    # === 4. 多工具切换 ===
    print("\n4. 多工具快速切换")

    TOOLS = {
        "gripper": {"position": [0.0, 0.0, 0.15]},
        "camera":  {"position": [0.0, 0.05, 0.08]},
        "suction": {"position": [0.0, 0.0, 0.10], "rotation": [0.0, math.pi/4, 0.0]},
    }

    for name, tcp_args in TOOLS.items():
        arm.set_tcp(**tcp_args)
        print(f"   切换到 '{name}': {tcp_args}")
        # 同一个 movel 目标, TCP 不同则法兰面位置不同
        # arm.movel(position=[0.3, 0.0, 0.15, 0.0, 0.0, 0.0], duration=1.5)

    # === 5. 重置 TCP (回到法兰面) ===
    print("\n5. 重置 TCP (法兰面为参考)")
    arm.clear_tcp()
    print("   TCP 已重置，运动指令以法兰面为参考")

    # === 6. TCP 的坐标变换公式 ===
    print("\n6. 坐标变换关系:")
    print()
    print("   T_base_tcp = T_base_flange × T_flange_tcp")
    print()
    print("   其中:")
    print("   T_base_flange = FK(joints)        ← 正运动学计算")
    print("   T_flange_tcp  = set_tcp() 设定    ← 用户配置")
    print("   T_base_tcp    = arm.get_pose()    ← SDK 返回 (含 TCP 偏移)")
    print()
    print("   知道任意两个, 可以求第三个:")
    print("   T_flange_tcp  = T_base_flange⁻¹ × T_base_tcp")
    print("   T_base_flange = T_base_tcp × T_flange_tcp⁻¹")

finally:
    arm.disable()
