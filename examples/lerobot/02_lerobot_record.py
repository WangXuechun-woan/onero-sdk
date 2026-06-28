"""
LeRobot 数据采集 — 录制 episodes 用于策略训练。

通过遥操作采集训练数据，保存为 LeRobot 标准格式 (HuggingFace Dataset)。
采集的数据可以直接用于训练 ACT、Diffusion Policy 等算法。

前提:
    pip install -e lerobot/lerobot_robot_onero
    pip install -e lerobot/lerobot_teleoperator_onero

运行 (命令行 — 推荐):
    lerobot-record --robot.type=onero --robot.can_channel=can0 \\
        --teleop.type=onero_teleop --teleop.leader_channel=can1 \\
        --fps=50 \\
        --repo-id=your-username/onero_pick_place \\
        --num-episodes=50

运行 (Python):
    python examples/lerobot/02_lerobot_record.py
"""

import time

from lerobot_robot_onero import OneroRobot, OneroRobotConfig
from lerobot_teleoperator_onero import OneroTeleop, OneroTeleopConfig

# === 配置 ===
ROBOT_CHANNEL = "can0"
LEADER_CHANNEL = "can1"
FPS = 50
NUM_EPISODES = 10
EPISODE_DURATION = 10.0  # 每个 episode 最长 10 秒

# 创建设备
robot = OneroRobot(OneroRobotConfig(can_channel=ROBOT_CHANNEL))
teleop = OneroTeleop(OneroTeleopConfig(leader_channel=LEADER_CHANNEL))

robot.connect()
teleop.connect()

print(f"数据采集: {NUM_EPISODES} episodes, {FPS} Hz")
print("每个 episode 开始时按 Enter，结束时按 Enter。\n")

episodes = []
dt = 1.0 / FPS

try:
    for ep in range(NUM_EPISODES):
        input(f"Episode {ep+1}/{NUM_EPISODES} — 按 Enter 开始录制...")
        print("  录制中... (按 Enter 结束)")

        frames = []
        t0 = time.monotonic()
        recording = True

        # 非阻塞检查 Enter 的简化版本 — 实际用 lerobot-record 更方便
        while time.monotonic() - t0 < EPISODE_DURATION:
            t_start = time.monotonic()

            # 读取 leader 动作
            action = teleop.get_action()

            # 执行并观察
            robot.send_action(action)
            obs = robot.get_observation()

            frames.append({"observation": obs, "action": action})

            # 维持帧率
            elapsed = time.monotonic() - t_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

        episodes.append(frames)
        print(f"  完成! {len(frames)} 帧 ({len(frames)/FPS:.1f}s)")

    print(f"\n采集完成! 共 {len(episodes)} episodes")
    print("\n提示: 使用 lerobot-record CLI 可以自动保存为 HuggingFace Dataset 格式:")
    print("  lerobot-record --robot.type=onero --repo-id=user/dataset_name")

finally:
    teleop.disconnect()
    robot.disconnect()
