"""
LeRobot 遥操作 — 双臂 Leader-Follower 控制。

使用两台 A1 机械臂:
  - Leader (主手, can1): freedrive 模式，操作员手动引导
  - Follower (从手, can0): 实时跟随 leader 的关节角度

这是 LeRobot 生态最基础的能力 — 通过遥操作采集数据。

前提:
    pip install -e lerobot/lerobot_robot_onero
    pip install -e lerobot/lerobot_teleoperator_onero

运行 (命令行):
    lerobot-teleoperate --robot.type=onero --robot.can_channel=can0 \\
        --teleop.type=onero_teleop --teleop.leader_channel=can1

运行 (Python):
    python examples/lerobot/01_lerobot_teleoperate.py
"""

from lerobot_robot_onero import OneroRobot, OneroRobotConfig
from lerobot_teleoperator_onero import OneroTeleop, OneroTeleopConfig

# 配置
robot_config = OneroRobotConfig(can_channel="can0", model="A1R")
teleop_config = OneroTeleopConfig(leader_channel="can1", model="A1R", damping=0.3)

# 创建设备
robot = OneroRobot(robot_config)
teleop = OneroTeleop(teleop_config)

# 连接
robot.connect()
teleop.connect()

print("遥操作已启动!")
print("移动 leader 臂 (can1)，follower 臂 (can0) 会实时跟随。")
print("按 Ctrl+C 退出。")

try:
    while True:
        # 从 leader 读取动作
        action = teleop.get_action()

        # 发送到 follower
        robot.send_action(action)

        # 读取 follower 状态 (可用于记录)
        obs = robot.get_observation()

except KeyboardInterrupt:
    print("\n停止遥操作")

finally:
    teleop.disconnect()
    robot.disconnect()
    print("已断开连接")
