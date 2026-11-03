"""可配置负载举升演示（默认 MuJoCo 仿真，--real 切换真机）。

仿真：
    pip install -e ".[sim,dynamics]"
    python examples/12_payload_lift.py --mass 1 --cog 0 0 0.10 --mounting humanoid_right

真机（A1R 右肩垂直安装）：
    python examples/12_payload_lift.py --real --mass 1 --cog 0 0 0.10 --mounting humanoid_right

mass 为工具与工件总质量 (kg)，cog 为其在末端坐标系中的重心 (m)。
"""

import argparse
import logging
import math
import os
import sys

# ROS Humble's Pinocchio conflicts with the venv Pinocchio/NumPy ABI.
env = os.environ.copy()
restart = False
for name in ("PYTHONPATH", "LD_LIBRARY_PATH"):
    paths = env.get(name, "").split(os.pathsep)
    cleaned = os.pathsep.join(path for path in paths if not path.startswith("/opt/ros"))
    if cleaned != env.get(name, ""):
        env[name] = cleaned
        restart = True
if restart:
    os.execve(sys.executable, [sys.executable, *sys.argv], env)

from onero_sdk import Arm, MountingPreset

# 负载演示关心最终误差，不重复打印 movej 过程警告。
logging.getLogger("onero_sdk.arm").addFilter(
    lambda record: "did not reach target within motion window" not in record.getMessage()
)


parser = argparse.ArgumentParser(description="A1 可配置负载举升演示")
parser.add_argument("--mass", type=float, required=True, help="负载总质量 (kg)")
parser.add_argument(
    "--cog", type=float, nargs=3, default=[0, 0, 0],
    metavar=("X", "Y", "Z"), help="负载重心 (m)",
)
parser.add_argument("--duration", type=float, default=4.0, help="每段运动时间 (s)")
parser.add_argument("--speed", type=float, default=1.0, help="运动倍速（0.5=半速，2=两倍速）")
parser.add_argument("--real", action="store_true", help="连接 CAN 真机（默认仿真）")
parser.add_argument("--channel", default="can0", help="SocketCAN 接口")
parser.add_argument("--model", choices=["A1L", "A1R"], default="A1R")
parser.add_argument(
    "--mounting",
    choices=[m.value for m in MountingPreset if m is not MountingPreset.CUSTOM],
    required=True,
    help="实体安装方向；A1L/A1R 肩装通常为 humanoid_left/right",
)
args = parser.parse_args()

if (not all(map(math.isfinite, [args.mass, args.duration, args.speed, *args.cog]))
        or args.mass < 0 or args.duration <= 0 or args.speed <= 0):
    parser.error("负载参数必须为有限数，--mass 不能为负，"
                 "--duration 和 --speed 必须大于 0")
motion_duration = args.duration / args.speed

poses = [
    ("收拢", [0.0, -0.5, 0.3, -1.0, 0.0, 0.5, 0.0]),
    ("伸展承载", [0.3, -0.8, 0.5, -0.5, 0.2, 0.3, 0.0]),
    ("抬升", [-0.2, -0.3, 0.1, -1.2, -0.1, 0.8, 0.0]),
    ("收回", [0.0, -0.5, 0.3, -1.0, 0.0, 0.5, 0.0]),
]
if args.model == "A1L":  # A1L 的关节方向与 A1R 镜像
    poses = [(name, [-q for q in joints]) for name, joints in poses]

# 人形肩装下镜像 world X，保持末端 Y/Z 不变。
x_mirror_signs = [-1, 1, -1, -1, -1, -1, 1]
poses += [
    (f"反向{name}", [q * sign for q, sign in zip(joints, x_mirror_signs)])
    for name, joints in poses
]

viewer = None
if args.real:
    arm = Arm(args.model, channel=args.channel, mounting=args.mounting)
else:
    try:
        import mujoco.viewer
        from onero_sdk.mujoco_backend import MujocoBackend
    except ImportError:
        raise SystemExit('仿真依赖未安装，请运行: pip install -e ".[sim,dynamics]"')

    backend = MujocoBackend(
        model_name=args.model, realtime=True, mounting=args.mounting,
        payload_mass=args.mass, payload_cog=args.cog,
    )
    viewer = mujoco.viewer.launch_passive(backend._model, backend._data)
    backend._on_step = viewer.sync
    arm = Arm(args.model, backend=backend, mounting=args.mounting)

clean_sim_exit = False
try:
    arm.set_payload(args.mass, args.cog)
    arm.set_stiffness("medium")  # 真机首次验证避免 stiff 高增益振荡
    mode = "真机" if args.real else "MuJoCo 仿真"
    print(f"{mode}: {args.mass:.2f} kg, 重心: {args.cog} m, "
          f"安装: {args.mounting}, 倍速: {args.speed:g}x, "
          f"重力(base): {arm.gravity_direction}")

    if args.real:
        shoulder_mounting = {
            "A1L": "humanoid_left", "A1R": "humanoid_right",
        }[args.model]
        if args.mounting != shoulder_mounting:
            print(f"警告: 若实体是 {args.model} 人形肩装，必须改用 "
                  f"--mounting {shoulder_mounting}；方向错误会产生反向重力补偿。")
        input("确认路径无障碍，按 Enter 执行（Ctrl+C 急停）...")
    else:
        print("注意: 仿真验证负载与轨迹，不包含现场障碍物碰撞。")

    try:
        initial = arm.enable()
        start_joints = list(initial.joints)
        for name, joints in poses:
            print(f"当前阶段：{name}")
            arm.movej(joints, duration=motion_duration)

        arm.movej(start_joints, duration=motion_duration)
        arm.disable()
        print("已回到本次运行的初始位置，电机已失能。")
        if not args.real:
            input("按 Enter 关闭仿真窗口...")
    except BaseException:
        if args.real and arm.enabled:
            arm.emergency_stop()
            print("真机已急停；电机已失能，请继续支撑机械臂。")
        raise
    clean_sim_exit = viewer is not None
finally:
    if arm.enabled and not args.real:
        arm.disable()
    arm.close()
    if viewer is not None:
        viewer.close()

if clean_sim_exit:
    # Avoid a MuJoCo/GLFW double-destruction crash during interpreter shutdown.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
