"""
LeRobot 数据回放 — 验证采集质量。

加载之前采集的 dataset，在真机上回放每个 episode，
验证数据采集是否正确、动作是否流畅。

前提:
    pip install -e lerobot/lerobot_robot_onero
    # 已有采集的 dataset (通过 lerobot-record 创建)

运行:
    python examples/lerobot/03_lerobot_replay.py --dataset=user/onero_pick_place

或命令行:
    lerobot-replay --robot.type=onero --robot.can_channel=can0 \\
        --repo-id=user/onero_pick_place --episode=0
"""

import argparse
import time

from lerobot_robot_onero import OneroRobot, OneroRobotConfig

# 解析参数
parser = argparse.ArgumentParser(description="Replay LeRobot dataset on Onero arm")
parser.add_argument("--dataset", type=str, default="user/onero_pick_place",
                    help="HuggingFace dataset ID")
parser.add_argument("--episode", type=int, default=0,
                    help="Episode index to replay")
parser.add_argument("--can-channel", type=str, default="can0")
parser.add_argument("--fps", type=float, default=50.0)
args = parser.parse_args()

# 加载 dataset
print(f"加载 dataset: {args.dataset}")
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset(args.dataset)
    print(f"  Episodes: {dataset.num_episodes}")
    print(f"  Total frames: {len(dataset)}")
except ImportError:
    print("错误: 请安装 lerobot: pip install lerobot")
    exit(1)
except Exception as e:
    print(f"错误: 无法加载 dataset — {e}")
    print("\n提示: 先用 lerobot-record 采集数据:")
    print("  lerobot-record --robot.type=onero --repo-id=user/onero_pick_place")
    exit(1)

# 获取 episode 数据
episode_data = dataset.get_episode(args.episode)
print(f"\n回放 Episode {args.episode}: {len(episode_data)} 帧")

# 连接机械臂
robot = OneroRobot(OneroRobotConfig(can_channel=args.can_channel))
robot.connect()

dt = 1.0 / args.fps

try:
    input("按 Enter 开始回放...")

    for i, frame in enumerate(episode_data):
        t0 = time.monotonic()

        # 提取 action 并发送
        action = {k: v for k, v in frame.items() if k in robot.action_features}
        robot.send_action(action)

        if i % int(args.fps) == 0:
            print(f"  帧 {i}/{len(episode_data)}")

        elapsed = time.monotonic() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)

    print("回放完成!")

finally:
    robot.disconnect()
