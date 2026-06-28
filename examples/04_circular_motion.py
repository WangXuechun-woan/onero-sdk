"""
圆弧运动 (movec) — 弧形擦拭、弧形焊接等场景。

movec 通过 3 点定义圆弧: 起点(当前位置) → 经过点 → 终点。
适用于：擦拭桌面、弧形焊缝、圆形装配等。

运行:
    python examples/04_circular_motion.py
"""

import numpy as np
from onero_sdk import Arm

arm = Arm("A1R", channel="can0")
arm.enable()

try:
    # 先移动到起始位姿 (位置 + 旋转向量, 6D)
    arm.movel(position=[0.3, -0.1, 0.2, 0.0, 0.0, 0.0], duration=2.0)
    print("到达起点")

    # 圆弧运动: 经过点 → 终点 (6D: [x, y, z, rx, ry, rz])
    via_pose = [0.35, 0.0, 0.25, 0.0, 0.0, 0.0]     # 经过点 (圆弧最高点)
    target_pose = [0.3, 0.1, 0.2, 0.0, 0.0, 0.0]    # 终点

    arm.movec(via_pose, target_pose, speed=0.1)
    print("圆弧完成!")

    # 也可以用 duration 控制时间
    arm.movec(
        via_pose=[0.35, 0.0, 0.25, 0.0, 0.0, 0.0],
        target_pose=[0.3, -0.1, 0.2, 0.0, 0.0, 0.0],
        duration=3.0,
    )
    print("反向圆弧完成!")

finally:
    arm.disable()
