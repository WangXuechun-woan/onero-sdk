"""
实时伺服控制 — 外部规划器/传感器驱动。

servoj 是最底层的高频命令接口。用户在自己的循环中
每帧给出目标关节角，SDK 执行安全检查并发送。

典型用途:
  - 外部轨迹规划器（MoveIt、Ruckig、TOPP-RA）输出
  - 视觉伺服（相机 → 控制目标）
  - 遥操作（主手 → 从手映射）
  - 自定义插值算法

重要: 必须以控制频率(100~200Hz)持续调用！
      如果停止调用，机械臂保持最后一帧目标。

运行:
    python examples/09_realtime_servo.py
"""

import time
import math
from onero_sdk import Arm

arm = Arm("A1R", channel="can0")
arm.enable()

freq = 100.0   # 控制频率 (Hz)
dt = 1.0 / freq

try:
    # === 示例: 关节 2 做正弦摆动 ===
    print("正弦摆动 5 秒...")
    t0 = time.monotonic()

    while True:
        elapsed = time.monotonic() - t0
        if elapsed > 5.0:
            break

        # 计算目标（你的规划器输出）
        target = [0.0] * 7
        target[1] = -0.5 * math.sin(2.0 * math.pi * 0.5 * elapsed)

        # 发送单帧命令
        arm.servoj(target)

        # 维持控制频率
        time.sleep(max(0, dt - (time.monotonic() - t0 - elapsed)))

    print("完成!")

    # === 带力前馈的 servoj ===
    print("\n带力前馈 (更好的跟踪)...")
    t0 = time.monotonic()

    while True:
        elapsed = time.monotonic() - t0
        if elapsed > 3.0:
            break

        target = [0.0] * 7
        target[1] = -0.3 * math.sin(2.0 * math.pi * 1.0 * elapsed)

        # tau_ff: 前馈力矩（可以是重力补偿或动力学模型输出）
        arm.servoj(target, tau_ff=[0.0] * 7)

        time.sleep(max(0, dt - (time.monotonic() - t0 - elapsed)))

    print("完成!")

finally:
    arm.disable()
