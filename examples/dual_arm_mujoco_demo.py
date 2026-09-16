"""
双臂 MuJoCo 仿真演示 — 通过公开 DualArm/Arm 控制链执行。

运行方式:
    python examples/dual_arm_mujoco_demo.py

说明:
- 使用当前 A1L/A1R URDF 与 humanoid_left/right 安装矩阵。
- 对称动作满足 q_left = -q_right。
- 所有命令经过 DualArm -> Arm -> SafetyBackend -> MuJoCo ArmBackend。
- 演示轨迹经过桌面避碰验证；运行时一旦检测到桌面接触即停止。
- SDK 本身不提供任意环境的通用碰撞规划，应用仍需负责工作空间避障。
"""
import sys

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, ".")

from onero_sdk import DualArm
from onero_sdk.dual_mujoco_backend import DualMujocoBackend
from onero_sdk.safety import SafetyConfig


def _has_table_contact(scene: DualMujocoBackend) -> bool:
    table_id = mujoco.mj_name2id(
        scene._model, mujoco.mjtObj.mjOBJ_BODY, "table")
    for index in range(scene._data.ncon):
        contact = scene._data.contact[index]
        body1 = scene._model.geom_bodyid[int(contact.geom1)]
        body2 = scene._model.geom_bodyid[int(contact.geom2)]
        if table_id in (body1, body2):
            return True
    return False


def main():
    print("=" * 60)
    print("  双臂 MuJoCo 仿真演示（公开 DualArm 控制链）")
    print("=" * 60)

    scene = DualMujocoBackend(model_name="A1R", realtime=True)
    left_backend, right_backend = scene.arm_backends()
    dual = DualArm.create(
        left_backend=left_backend,
        right_backend=right_backend,
        arm_spacing=0.24,
        # MuJoCo 简化执行器可能滞后于命令参考。仅对仿真关闭硬件
        # tracking-error latch，其余关节/速度/限位保护保持开启。
        safety=SafetyConfig(
            tracking_error_hard_threshold=0,
            tracking_error_soft_threshold=0,
        ),
    )
    dual.enable()

    viewer = mujoco.viewer.launch_passive(scene._model, scene._data)
    viewer.cam.azimuth = -150
    viewer.cam.elevation = -20
    viewer.cam.distance = 2.0
    viewer.cam.lookat[:] = [0.0, 0.0, 0.9]
    viewer.sync()

    right_current = np.zeros(7)
    left_current = np.zeros(7)

    def move_pair(right_target, left_target, duration=2.0):
        nonlocal right_current, left_current
        right_target = np.asarray(right_target, dtype=float)
        left_target = np.asarray(left_target, dtype=float)
        steps = max(1, int(duration / scene._control_dt))
        for step in range(1, steps + 1):
            u = step / steps
            blend = u * u * (3.0 - 2.0 * u)
            right_command = right_current + blend * (right_target - right_current)
            left_command = left_current + blend * (left_target - left_current)
            dual.servoj(left_command.tolist(), right_command.tolist())
            if _has_table_contact(scene):
                raise RuntimeError("demo trajectory contacted the table")
            viewer.sync()  # Viewer/OpenGL stays on the main thread.
        right_current = right_target
        left_current = left_target

    safe_right_poses = [
        np.array([0.25, -0.10, 0.10, 0.30, 0.05, 0.20, 0.00]),
        np.array([-0.45, -0.20, 0.20, 0.55, 0.10, 0.30, 0.05]),
        np.array([0.00, 0.00, -0.30, 0.30, -0.20, 0.20, 0.00]),
        np.zeros(7),
    ]

    try:
        input("按 Enter 开始镜像协调演示...")
        for index, right_target in enumerate(safe_right_poses, start=1):
            print(f"Scene {index}: 镜像姿态")
            move_pair(right_target, -right_target)
        print("演示完成。整个轨迹未检测到桌面接触。")
        input("按 Enter 关闭...")
    finally:
        # 仿真可安全显式失能。真机只应在机械臂得到可靠支撑后执行。
        dual.disable()
        dual.left.close()
        dual.right.close()
        viewer.close()


if __name__ == "__main__":
    main()
