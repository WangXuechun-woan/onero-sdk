"""
刚度切换 — 在运行中调整机械臂的软硬程度。

三种预设:
  - STIFF:  硬，精确定位（装配、焊接）
  - MEDIUM: 中等刚度（常规搬运）
  - SOFT:   柔软，安全（人机交互、教学）

也可以直接传数值 kp/kd 精确控制。

运行:
    python examples/21_stiffness.py

真机注意：手动推动前请确认运动范围内无人、无障碍物，并准备好急停；
调用 disable() 会使能器失能，失能前必须可靠支撑机械臂。
"""

import time

from onero_sdk import Arm, Stiffness

arm = Arm("A1R", channel="can0")
arm.enable()

target = [0.0, -0.5, 0.3, -1.0, 0.0, 0.5, 0.0]

try:
    for stiffness, label in (
        (Stiffness.STIFF, "STIFF"),
        (Stiffness.MEDIUM, "MEDIUM"),
        (Stiffness.SOFT, "SOFT"),
    ):
        arm.set_stiffness(stiffness)
        # 重新发送同一目标，使当前刚度成为最终保持增益。
        arm.movej(target, duration=7.0)
        print(f"{label} 模式：请手动推动机械臂，观察保持力；5 秒后切换")
        time.sleep(5.0)

finally:
    # 真机失能前请确认机械臂已有可靠支撑；disable() 是失能，不是普通停止。
    arm.disable()
