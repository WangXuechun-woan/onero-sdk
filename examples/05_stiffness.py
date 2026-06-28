"""
刚度切换 — 在运行中调整机械臂的软硬程度。

三种预设:
  - STIFF:  硬，精确定位（装配、焊接）
  - MEDIUM: 中等刚度（常规搬运）
  - SOFT:   柔软，安全（人机交互、教学）

也可以直接传数值 kp/kd 精确控制。

运行:
    python examples/05_stiffness.py
"""

from onero_sdk import Arm, Stiffness

arm = Arm("A1R", channel="can0")
arm.enable()

target = [0.0, -0.5, 0.3, -1.0, 0.0, 0.5, 0.0]

try:
    # === 高刚度: 精确到位 ===
    arm.set_stiffness(Stiffness.STIFF)
    arm.movej(target, duration=2.0)
    print(f"STIFF 模式: 臂很硬，手推不动")

    # === 中等刚度: 普通任务 ===
    arm.set_stiffness(Stiffness.MEDIUM)
    arm.movej(target, duration=2.0)
    print(f"MEDIUM 模式: 有一定柔性")

    # === 低刚度: 人机安全 ===
    arm.set_stiffness(Stiffness.SOFT)
    arm.movej(target, duration=2.0)
    print(f"SOFT 模式: 很柔软，可以用手拨动")

    # === 直接指定 kp / kd ===
    arm.movej(target, duration=2.0, kp=30.0, kd=0.8)
    print(f"自定义 kp=30, kd=0.8")

finally:
    arm.disable()
