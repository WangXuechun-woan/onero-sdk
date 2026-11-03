"""真机首次小幅运动。运行前清空工作空间并确认急停可用。"""
from onero_sdk import Arm, OneroError

SAFE_TARGET = [0.0, -0.15, 0.0, -0.15, 0.0, 0.0, 0.0]
arm = Arm("A1R", channel="can0")
try:
    arm.enable()
    print("起始关节角：", [round(value, 3) for value in arm.state.joints])
    input("确认工作空间安全后按 Enter 开始小幅运动，Ctrl+C 取消：")
    arm.movej(SAFE_TARGET, duration=7.0)
    print("动作完成。")
except (KeyboardInterrupt, OneroError) as exc:
    if arm.enabled:
        arm.stop(deceleration=2.0)
    print(f"动作已停止：{exc}")
finally:
    # 失能会撤销保持力矩；确认机械臂受支撑后再取消下一行注释。
    # arm.disable()
    arm.close()
