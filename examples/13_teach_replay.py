"""
示教回放 — 手动引导 + 自动重复执行。

工厂中最常见的编程方式:
1. 手动引导机械臂走一遍（freedrive/示教模式）
2. 记录轨迹
3. 重复回放

也可以将轨迹保存到文件，下次启动直接加载。

运行:
    python examples/13_teach_replay.py
"""

import time
from onero_sdk import Arm

arm = Arm("A1R", channel="can0")
arm.enable()

try:
    # === 第 1 步: 进入示教模式 ===
    print("进入示教模式 — 请手动引导机械臂...")
    print("(10 秒后自动退出)")
    arm.start_recording()
    arm.freedrive_enable(damping=0.5)

    time.sleep(10.0)  # 给用户 10 秒引导时间

    arm.freedrive_disable()
    recording = arm.stop_recording()
    print(f"录制完成! 共 {recording.num_frames} 帧, "
          f"时长 {recording.duration:.1f} 秒")

    # === 第 2 步: 回放 ===
    print("\n回放轨迹 (原速)...")
    arm.replay(recording)
    print("回放完成!")

    # === 第 3 步: 不同速度回放 ===
    print("\n2 倍速回放...")
    arm.replay(recording, speed_factor=2.0)
    print("完成!")

    print("\n0.5 倍速（慢动作）...")
    arm.replay(recording, speed_factor=0.5)
    print("完成!")

    # === 第 4 步: 保存到文件 ===
    recording.save("my_trajectory.json")
    print("\n轨迹已保存到 my_trajectory.json")

    # === 第 5 步: 从文件加载 ===
    from onero_sdk.trajectory import Recording
    loaded = Recording.load("my_trajectory.json")
    print(f"从文件加载: {loaded.num_frames} 帧")
    arm.replay(loaded)
    print("加载并回放完成!")

finally:
    arm.disable()
