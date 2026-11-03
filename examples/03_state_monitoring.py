"""
读取状态与监控 — 了解机械臂当前在做什么。

演示如何读取关节角度、速度、力矩，以及如何判断运动状态。
适用于：状态记录、安全监控、条件触发等场景。

运行:
    python examples/03_state_monitoring.py
"""

import time
from onero_sdk import Arm

arm = Arm("A1R", channel="can0")
arm.enable()

try:
    # === 1. 读取当前状态 ===
    state = arm.get_state()
    print("当前状态:")
    print(f"  关节角度 (rad): {[f'{j:.3f}' for j in state.joints]}")
    print(f"  关节速度 (rad/s): {[f'{v:.3f}' for v in state.velocities]}")
    print(f"  关节力矩 (Nm): {[f'{t:.3f}' for t in state.torques]}")
    print(f"  时间戳: {state.timestamp:.3f}")

    # === 2. 获取关节数量 ===
    print(f"\n关节数: {arm.num_joints}")

    # === 3. 检查是否在运动中 ===
    print(f"动作开始前是否在运动: {arm.is_moving}")

    # === 4. 运动中实时监控 ===
    import threading

    def monitor(done):
        """后台线程打印动作期间的实时状态"""
        while not done.is_set():
            s = arm.get_state()
            pos = s.joints[1]  # 监控第 2 个关节
            vel = s.velocities[1]
            moving = any(abs(value) > 0.01 for value in s.velocities)
            # print(f"  是否在运动: {moving}; J2: pos={pos:.3f} rad, vel={vel:.3f} rad/s")
            time.sleep(0.1)

    print("\n运动中实时监控 (J2):")
    done = threading.Event()
    t = threading.Thread(target=monitor, args=(done,), daemon=True)
    t.start()

    try:
        arm.movej([0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0], duration=3.0)
    finally:
        done.set()
    t.join()

    # === 5. 读取并显示末端位姿 (需要 IK 依赖) ===
    try:
        pos, rot = arm.get_pose()
        print(f"\n末端位置: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f} (m)")
    except Exception:
        print("\n(安装 onero-sdk[ik] 可查看末端位姿)")

finally:
    arm.disable()
