"""
安全停止 — 从另一个线程中断正在执行的运动。

实际应用中，常需要在检测到异常时立即停止机械臂，
例如: 碰撞检测、用户按下急停按钮、外部传感器触发等。

运行:
    python examples/07_safe_stop.py
"""

import time
import threading
from onero_sdk import Arm

arm = Arm("A1", channel="can0")
arm.enable()

try:
    # === 场景: 运动过程中被中断 ===

    def emergency_stop_after(seconds):
        """模拟: N 秒后触发急停"""
        time.sleep(seconds)
        print(f"\n  [!] {seconds}s 后触发停止!")
        arm.stop(deceleration=3.0)  # 从另一个线程调用
        print("  [!] 已减速停止")

    # 启动一个 5 秒的长运动
    print("开始 5 秒长运动...")
    t = threading.Thread(target=emergency_stop_after, args=(2.0,), daemon=True)
    t.start()

    arm.movej([0.0, -1.5, 0.5, -1.0, 0.0, 0.5, 0.0], duration=5.0)
    t.join()

    print(f"  停在: J2={arm.state.joints[1]:.3f} rad (目标是 -1.5)")
    print("  ✓ 运动被安全中断，没有跑完全程")

    # === 正确的急停模式 ===
    print("""
推荐的急停模式:

    import threading
    from onero_sdk import Arm

    arm = Arm("A1", channel="can0")
    arm.enable()

    # 方式 1: arm.stop() — 减速停止（推荐）
    # 从任何线程调用，会中断当前运动并平滑减速
    arm.stop(deceleration=3.0)

    # 方式 2: arm.disable() — 立即断电（紧急情况）
    # 电机立即失去力矩，机械臂靠重力下垂
    arm.disable()
""")

finally:
    arm.disable()
