"""
LeRobot 策略推理 — 用训练好的模型控制机械臂。

加载训练好的策略 (ACT / Diffusion Policy / VLA)，
在真机上执行自主操作。

支持的算法:
  - ACT (Action Chunking with Transformers)
  - Diffusion Policy
  - TDMPC
  - 其他 LeRobot 兼容的 policy

前提:
    pip install -e lerobot/lerobot_robot_onero
    # 已训练好的模型 checkpoint

运行:
    python examples/lerobot/04_lerobot_policy.py \\
        --policy=act \\
        --ckpt=outputs/train/act_onero_pick/checkpoints/last/pretrained_model

或命令行:
    lerobot-infer --robot.type=onero --robot.can_channel=can0 \\
        --policy.type=act \\
        --policy.path=outputs/train/act_onero_pick/checkpoints/last/pretrained_model
"""

import argparse
import time

import torch
from lerobot_robot_onero import OneroRobot, OneroRobotConfig

parser = argparse.ArgumentParser(description="Run trained policy on Onero arm")
parser.add_argument("--policy", type=str, default="act",
                    choices=["act", "diffusion", "tdmpc", "vla"],
                    help="Policy type")
parser.add_argument("--ckpt", type=str, required=True,
                    help="Path to model checkpoint")
parser.add_argument("--can-channel", type=str, default="can0")
parser.add_argument("--fps", type=float, default=50.0)
parser.add_argument("--max-steps", type=int, default=500,
                    help="Max inference steps per episode")
args = parser.parse_args()

# === 加载策略 ===
print(f"加载 {args.policy} 策略: {args.ckpt}")

try:
    from lerobot.common.policies.factory import make_policy
    from omegaconf import OmegaConf

    policy = make_policy(args.policy, checkpoint=args.ckpt)
    policy.eval()
    if torch.cuda.is_available():
        policy = policy.cuda()
        print("  使用 GPU 推理")
    else:
        print("  使用 CPU 推理 (较慢)")
except ImportError as e:
    print(f"错误: {e}")
    print("\n请确保已安装 lerobot 及策略依赖:")
    print("  pip install lerobot[act]")
    exit(1)

# === 连接机械臂 ===
robot = OneroRobot(OneroRobotConfig(can_channel=args.can_channel))
robot.connect()

dt = 1.0 / args.fps

print(f"\n策略推理: {args.policy}, {args.fps} Hz, 最多 {args.max_steps} 步")
input("按 Enter 开始自主执行... (Ctrl+C 急停)")

try:
    for step in range(args.max_steps):
        t0 = time.monotonic()

        # 1. 获取观测
        obs = robot.get_observation()

        # 2. 策略推理
        with torch.no_grad():
            obs_tensor = policy.normalize_inputs(obs)
            action = policy.select_action(obs_tensor)

        # 3. 执行动作
        action_dict = policy.unnormalize_outputs(action)
        robot.send_action(action_dict)

        # 4. 帧率控制
        elapsed = time.monotonic() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)

        if step % int(args.fps) == 0:
            print(f"  步数: {step}/{args.max_steps}")

    print("执行完成!")

except KeyboardInterrupt:
    print("\n用户中断 — 停止")

finally:
    robot.disconnect()
    print("已断开")
