"""基于现场示教活动范围设计的双臂协调演示（默认 MuJoCo 仿真）。

仿真：
    python examples/18_dual_arm.py

仿真确认无误后运行真机：
    python examples/18_dual_arm.py --real --left-channel can0 --right-channel can1
"""

import argparse
import logging
import math
import os
import sys
import threading
import time
import traceback

import numpy as np

from onero_sdk import DualArm
from onero_sdk.motion_executor import MotionFrame


# 本演示会单独检查最终到位误差，不重复打印 movej 过程提示。
logging.getLogger("onero_sdk.arm").addFilter(
    lambda record: "did not reach target within motion window" not in record.getMessage()
)


# 根据 dual_arm_safe_path.json 的活动范围离线设计；运行时不读取轨迹文件。
MIRROR_CENTER_RIGHT = [1.0, -0.2, 0.2, 0.8, 0.0, 0.2, 0.0]
DEFAULT_ARM_SPACING = 0.12
HEART_START_WORLD_RIGHT = [0.41, -0.05, 0.05]
HEART_START_SEED_RIGHT = [1.133566, -0.097426, -0.779708, 1.338379,
                          0.433993, 0.201596, -0.319000]
PREPARE_DURATION = 6.0
PREPARE_TOLERANCE = 0.05  # rad
HEART_DURATION = 14.0
HEART_HALF_WIDTH = 0.14
HEART_HEIGHT = 0.26
MAX_JOINT_SPEED = math.radians(180.0)

# A1R 规格限位；A1L 的关节方向与 A1R 镜像。
RIGHT_JOINT_LIMITS = np.radians([
    [-60, 180], [-180, 15], [-158, 158], [-110, 110],
    [-158, 158], [-90, 90], [-158, 158],
])
LEFT_JOINT_LIMITS = np.column_stack(
    (-RIGHT_JOINT_LIMITS[:, 1], -RIGHT_JOINT_LIMITS[:, 0])
)


def mirrored_left(right):
    return [-joint for joint in right]


MIRROR_CENTER_LEFT = mirrored_left(MIRROR_CENTER_RIGHT)
# shortcut: 仿真只覆盖示教范围内的一个非对称起点，真机每次仍需人工确认接入路径。
SIM_START_LEFT = [-0.85, 0.10, -0.12, -0.90, 0.08, -0.25, 0.08]
SIM_START_RIGHT = [0.92, -0.26, 0.10, 0.92, 0.04, 0.12, -0.05]


def start_error(left_state, right_state, left_target, right_target):
    return max(
        *(abs(float(a) - b) for a, b in zip(left_state.joints, left_target)),
        *(abs(float(a) - b) for a, b in zip(right_state.joints, right_target)),
    )


def move_pair(dual, left, right, duration):
    # 固定步数的余弦轨迹保证共享双臂后端每周期严格成对。
    return dual._parallel(
        lambda: dual.left.movej(left, duration=duration, profile="cosine"),
        lambda: dual.right.movej(right, duration=duration, profile="cosine"),
    )


def heart_start_joints(dual):
    _, current_rot = dual.right.get_pose(frame="world")
    target_pos, _ = dual.right._input_pose_transform(
        HEART_START_WORLD_RIGHT, current_rot, frame="world")
    fk = dual.right._get_fk()
    _, target_rot = fk.get_ee_pose(HEART_START_SEED_RIGHT)
    ik = dual.right._get_ik()
    ik.reset(np.asarray(HEART_START_SEED_RIGHT, dtype=float))
    right = ik.solve(
        target_pos, target_rot, dt=1.0 / dual.right.control_freq,
        max_iter=200, pos_tol=1e-3, rot_tol=1e-2, strict=True,
    ).tolist()
    return mirrored_left(right), right


def prepare_demo(dual, speed, left_target, right_target):
    print("当前阶段：双臂并行到达爱心上方起点")
    move_pair(
        dual, left_target, right_target,
        PREPARE_DURATION / speed,
    )
    error = start_error(
        dual.left.state, dual.right.state, left_target, right_target)
    if error > PREPARE_TOLERANCE:
        raise RuntimeError(
            "双臂未准确到达爱心上方起点，"
            f"最大关节误差为 {error:.3f} rad"
        )
    print("双臂已到达爱心上方起点。")


def heart_joint_paths(dual, duration):
    control_hz = min(dual.left.control_freq, dual.right.control_freq)
    steps = max(2, round(duration * control_hz))
    start_pos, start_rot = dual.right.get_pose(frame="world")
    ik = dual.right._get_ik()
    ik.reset(np.asarray(dual.right.state.joints, dtype=float))
    right_path = []

    for step in range(steps + 1):
        angle = math.pi * step / steps
        horizontal = 16.0 * math.sin(angle) ** 3
        vertical = (
            13.0 * math.cos(angle)
            - 5.0 * math.cos(2.0 * angle)
            - 2.0 * math.cos(3.0 * angle)
            - math.cos(4.0 * angle)
        )
        target_world = start_pos + np.array([
            0.0,
            -HEART_HALF_WIDTH * horizontal / 16.0,
            HEART_HEIGHT * (vertical - 5.0) / 28.923,
        ])
        target_pos, target_rot = dual.right._input_pose_transform(
            target_world, start_rot, frame="world")
        joints = ik.solve(
            target_pos, target_rot, dt=1.0 / control_hz,
            max_iter=100, pos_tol=1e-3, rot_tol=1e-2, strict=True,
        )
        right_path.append(np.asarray(joints, dtype=float))

    right_path = np.asarray(right_path)
    left_path = -right_path
    for side, path, limits in (
        ("左臂", left_path, LEFT_JOINT_LIMITS),
        ("右臂", right_path, RIGHT_JOINT_LIMITS),
    ):
        invalid = np.argwhere(
            (path < limits[:, 0] - 1e-6) | (path > limits[:, 1] + 1e-6)
        )
        if invalid.size:
            point, joint = invalid[0]
            raise RuntimeError(
                f"{side}心形轨迹第 {point} 点的 J{joint + 1} "
                "超出规格限位"
            )

    sample_dt = duration / steps
    max_speed = max(
        float(np.max(np.abs(np.diff(left_path, axis=0)))) / sample_dt,
        float(np.max(np.abs(np.diff(right_path, axis=0)))) / sample_dt,
    )
    if max_speed > MAX_JOINT_SPEED + 1e-6:
        raise RuntimeError(
            f"心形轨迹最大关节速度 {math.degrees(max_speed):.1f}°/s "
            "超过 180°/s"
        )
    return left_path, right_path, sample_dt


def stream_joint_paths(dual, left_path, right_path, sample_dt):
    """用与 movej 相同的规划执行器连续发送两臂轨迹。"""
    velocities = (
        np.gradient(left_path, sample_dt, axis=0),
        np.gradient(right_path, sample_dt, axis=0),
    )
    velocities[0][[0, -1]] = 0.0
    velocities[1][[0, -1]] = 0.0
    start = threading.Barrier(2)
    stop = threading.Event()
    errors = []

    def run(arm, path, velocity_path):
        try:
            arm._check_enabled()
            arm._check_no_freedrive()
            arm._begin_motion()
            try:
                start.wait()

                def frames():
                    for joints, joint_velocity in zip(path, velocity_path):
                        if stop.is_set():
                            return
                        kps, kds = arm._compute_gains(joints)
                        yield MotionFrame(
                            positions=joints.tolist(),
                            velocities=joint_velocity.tolist(),
                            kps=kps,
                            kds=kds,
                            torques=[0.0] * arm.num_joints,
                            dt=sample_dt,
                        )

                arm._state = arm._executor.run(frames())
            finally:
                arm._end_motion()
        except BaseException as exc:
            stop.set()
            try:
                start.abort()
            except threading.BrokenBarrierError:
                pass
            errors.append(exc)

    workers = [
        threading.Thread(target=run, args=(dual.left, left_path, velocities[0])),
        threading.Thread(target=run, args=(dual.right, right_path, velocities[1])),
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    if errors:
        raise errors[0]


def run_demo(dual, speed):
    duration = HEART_DURATION / speed
    print("正在预计算心形轨迹并检查关节限位与速度...")
    left_path, right_path, sample_dt = heart_joint_paths(dual, duration)
    print("当前阶段：左臂画左半边，右臂画右半边，共同组成一个爱心")
    stream_joint_paths(dual, left_path, right_path, sample_dt)

    print("双臂已在爱心下方尖点视觉汇合，并保留安全间距。")
    time.sleep(1.0 / speed)
    print("当前阶段：双臂镜像回到工作中心")
    move_pair(
        dual, MIRROR_CENTER_LEFT, MIRROR_CENTER_RIGHT,
        2.0 / speed,
    )
    print("双臂心形协调演示完成，已返回镜像工作中心。")


def main():
    parser = argparse.ArgumentParser(description="双臂现场安全范围协调演示")
    parser.add_argument("--real", action="store_true", help="连接 CAN 真机（默认仿真）")
    parser.add_argument("--left-channel", default="can0", help="左臂 SocketCAN 接口")
    parser.add_argument("--right-channel", default="can1", help="右臂 SocketCAN 接口")
    parser.add_argument("--arm-spacing", type=float, default=DEFAULT_ARM_SPACING,
                        help="双臂底座中心距 (m)，默认 0.12")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="演示倍速；真机最大为 1，仿真最大为 4")
    args = parser.parse_args()
    if (not math.isfinite(args.speed) or args.speed <= 0
            or args.speed > (1.0 if args.real else 4.0)):
        parser.error("--speed 必须大于 0；真机最大 1，仿真最大 4")
    if not math.isfinite(args.arm_spacing) or not 0.10 <= args.arm_spacing <= 0.30:
        parser.error("--arm-spacing 必须是 0.10～0.30 m 的有限数")

    viewer = None
    scene = None
    if args.real:
        input("确认桌面及其他障碍物未变化、急停可用；按 Enter 连接并使能真机...")
        dual = DualArm.create(
            left_channel=args.left_channel,
            right_channel=args.right_channel,
            arm_spacing=args.arm_spacing,
        )
    else:
        try:
            import mujoco
            import mujoco.viewer
            from onero_sdk.dual_mujoco_backend import DualMujocoBackend
            from onero_sdk.safety import SafetyConfig
        except ImportError:
            raise SystemExit('仿真依赖未安装，请运行: pip install -e ".[sim,dynamics]"')

        scene = DualMujocoBackend(model_name="A1R", realtime=True)
        for body_name, y in (
            ("R_base_link", -args.arm_spacing / 2.0),
            ("L_base_link", args.arm_spacing / 2.0),
        ):
            body_id = mujoco.mj_name2id(
                scene._model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            scene._model.body_pos[body_id, 1] = y
        table_id = mujoco.mj_name2id(
            scene._model, mujoco.mjtObj.mjOBJ_BODY, "table")
        # 现场桌面只在左臂下方；高度按示教活动范围留出仿真余量。
        scene._model.body_pos[table_id, 1] = 0.45
        scene._model.body_pos[table_id, 2] = 0.64
        scene.reset(right_pos=SIM_START_RIGHT, left_pos=SIM_START_LEFT)

        def reject_contact(shared):
            if shared._data.ncon:
                raise RuntimeError("新设计路线在仿真中检测到碰撞")

        scene.on_step = reject_contact
        left_backend, right_backend = scene.arm_backends()
        dual = DualArm.create(
            left_backend=left_backend,
            right_backend=right_backend,
            arm_spacing=args.arm_spacing,
            safety=SafetyConfig(
                tracking_error_hard_threshold=0,
                tracking_error_soft_threshold=0,
            ),
        )
        viewer = mujoco.viewer.launch_passive(scene._model, scene._data)
        viewer.cam.azimuth = -180
        viewer.cam.elevation = -10
        viewer.cam.distance = 2.0
        viewer.cam.lookat[:] = [0.1, 0.0, 0.9]
        viewer.sync()

    try:
        dual.enable()
        heart_start_left, heart_start_right = heart_start_joints(dual)
        if args.real:
            input(
                "确认两臂当前位置到爱心上方起点的关节路径无障碍；"
                "按 Enter 执行准备动作..."
            )
            prepare_demo(
                dual, args.speed, heart_start_left, heart_start_right)
            input("爱心起点已到达；确认路径内无人且桌面未移动，按 Enter 开始演示...")
            run_demo(dual, args.speed)
            print("真机保持镜像工作中心和使能状态；可靠支撑后再主动失能。")
        else:
            errors = []

            def simulate():
                try:
                    prepare_demo(
                        dual, args.speed, heart_start_left, heart_start_right)
                    run_demo(dual, args.speed)
                except BaseException as exc:
                    errors.append(exc)

            worker = threading.Thread(target=simulate)
            worker.start()
            while worker.is_alive() and viewer.is_running():
                with scene._command_condition:
                    viewer.sync()
                time.sleep(0.01)
            worker.join()
            if errors:
                raise errors[0]
            if viewer.is_running():
                with scene._command_condition:
                    viewer.sync()
                input("仿真完成且未检测到碰撞；确认动作无误后按 Enter 关闭窗口。")
    finally:
        if args.real:
            dual.left.close()
            dual.right.close()
        else:
            if dual.left.enabled and dual.right.enabled:
                dual.disable()
            dual.left.close()
            dual.right.close()
            if viewer is not None:
                viewer.close()
            exc_type, exc, exc_tb = sys.exc_info()
            if exc is not None:
                traceback.print_exception(exc_type, exc, exc_tb)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1 if exc is not None else 0)


if __name__ == "__main__":
    main()
