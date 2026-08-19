"""
启动 BEAVR VR 遥操作 Onero A1。

用法:
    # 默认配置 (侧装左臂, Meta Quest 3)
    python -m beavr_onero.run

    # 指定参数
    python -m beavr_onero.run \
        --can-channel=can0 \
        --mounting=side_mount_left \
        --vr-port=5555 \
        --scale=0.8

    # 右臂
    python -m beavr_onero.run --mounting=side_mount_right --can-channel=can1
"""

import argparse

from .robot_wrapper import OneroA1Config
from .operator import OneroA1Operator, OneroOperatorConfig
from onero_sdk.mounting import MountingPreset


def main():
    parser = argparse.ArgumentParser(
        description="BEAVR VR Teleoperation for Onero A1")
    parser.add_argument("--can-channel", type=str, default="can0",
                        help="CAN 通道 (default: can0)")
    parser.add_argument("--model", type=str, default="A1R",
                        help="机械臂型号")
    parser.add_argument("--mounting", type=str, default="side_mount_left",
                        choices=[m.value for m in MountingPreset],
                        help="安装方式 (default: side_mount_left)")
    parser.add_argument("--vr-port", type=int, default=5555,
                        help="VR ZMQ 订阅端口 (default: 5555)")
    parser.add_argument("--vr-topic", type=str, default="hand_right",
                        help="VR ZMQ topic (default: hand_right)")
    parser.add_argument("--host", type=str, default="localhost",
                        help="VR ZMQ host (default: localhost)")
    parser.add_argument("--control-freq", type=float, default=60.0,
                        help="控制频率 Hz (default: 60)")
    parser.add_argument("--scale", type=float, default=0.8,
                        help="位移缩放系数 (default: 0.8)")
    parser.add_argument("--scale-xyz", type=str, default=None,
                        help="各轴独立缩放, 逗号分隔 (如: 0.8,0.8,0.7)")
    parser.add_argument("--smoothing", type=float, default=0.3,
                        help="低通滤波系数 0-1 (default: 0.3)")
    parser.add_argument("--no-adaptive-smoothing", action="store_true",
                        help="禁用速度自适应滤波")
    parser.add_argument("--max-linear-speed", type=float, default=0.5,
                        help="最大末端线速度 m/s (default: 0.5)")
    parser.add_argument("--max-angular-speed", type=float, default=2.0,
                        help="最大末端角速度 rad/s (default: 2.0)")
    parser.add_argument("--max-frame-age", type=float, default=0.20,
                        help="VR 帧超时时间秒 (default: 0.20)")
    parser.add_argument("--workspace-min", type=str, default=None,
                        help="工作空间下界 x,y,z, 米 (如: 0.05,-0.4,0.02)")
    parser.add_argument("--workspace-max", type=str, default=None,
                        help="工作空间上界 x,y,z, 米 (如: 0.6,0.4,0.6)")
    parser.add_argument("--yaw-offset", type=float, default=0.0,
                        help="安装 yaw 偏移修正 (度)")
    args = parser.parse_args()

    # 解析缩放
    if args.scale_xyz:
        scale_xyz = [float(x) for x in args.scale_xyz.split(",")]
    else:
        scale_xyz = [args.scale] * 3
    workspace_min = [float(x) for x in args.workspace_min.split(",")] if args.workspace_min else None
    workspace_max = [float(x) for x in args.workspace_max.split(",")] if args.workspace_max else None

    # 构建配置
    robot_config = OneroA1Config(
        can_channel=args.can_channel,
        model=args.model,
        mounting=args.mounting,
        position_scale=args.scale,
        position_scale_xyz=scale_xyz,
        yaw_offset_deg=args.yaw_offset,
        control_freq=args.control_freq,
        smoothing_alpha=args.smoothing,
        adaptive_smoothing=not args.no_adaptive_smoothing,
        max_linear_speed=args.max_linear_speed,
        max_angular_speed=args.max_angular_speed,
    )
    if workspace_min is not None:
        robot_config.workspace_min = workspace_min
    if workspace_max is not None:
        robot_config.workspace_max = workspace_max

    operator_config = OneroOperatorConfig(
        host=args.host,
        vr_sub_port=args.vr_port,
        vr_topic=args.vr_topic,
        control_freq=args.control_freq,
        max_frame_age_s=args.max_frame_age,
        robot_config=robot_config,
    )

    # 打印配置
    mounting = MountingPreset(args.mounting)
    print("=" * 50)
    print(" BEAVR VR Teleoperation — Onero A1")
    print("=" * 50)
    print(f"  CAN:      {args.can_channel}")
    print(f"  安装方式: {mounting.value}")
    print(f"  VR 端口:  tcp://{args.host}:{args.vr_port} [{args.vr_topic}]")
    print(f"  缩放:     {scale_xyz}")
    print(f"  频率:     {args.control_freq} Hz")
    print(f"  滤波:     {args.smoothing} adaptive={not args.no_adaptive_smoothing}")
    print(f"  超时:     {args.max_frame_age}s")
    print("=" * 50)
    print()

    # 启动
    operator = OneroA1Operator(operator_config)
    operator.start()


if __name__ == "__main__":
    main()
