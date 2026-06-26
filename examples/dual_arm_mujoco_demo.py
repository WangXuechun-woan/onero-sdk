"""
双臂 MuJoCo 仿真演示 — 人体躯干 + 双臂协调动作。

运行方式:
    python examples/dual_arm_mujoco_demo.py

展示内容:
    1. 双臂自然下垂 → T-pose 展开
    2. 双臂前摆
    3. 双臂协调弯曲 (模拟搬运)
    4. 单臂挥手

关节符号约定 (与业界一致):
    反号 = 对称镜像运动 (L_joints = -R_joints → 两臂做对称动作)
    J1: 前后摆动 (opposite sign → both forward)
    J2: 侧向展开 (opposite sign → both outward / T-pose)
    J4: 肘部弯曲 (opposite sign → both bend)
"""
import sys
import numpy as np

sys.path.insert(0, '.')

from onero_sdk.dual_mujoco_backend import DualMujocoBackend
import mujoco.viewer


def mirror(joints):
    """Mirror right arm pose to left arm (negate all joints)."""
    return (-np.asarray(joints)).tolist()


def main():
    print("=" * 60)
    print("  双臂 MuJoCo 仿真演示")
    print("  人体躯干 + 左右臂协调动作")
    print("=" * 60)
    print()

    backend = DualMujocoBackend(model_name='A1', realtime=True)
    viewer = mujoco.viewer.launch_passive(backend._model, backend._data)
    backend._viewer = viewer

    viewer.cam.azimuth = 0
    viewer.cam.elevation = -15
    viewer.cam.distance = 2.2
    viewer.cam.lookat[:] = [0.2, 0, 0.85]
    viewer.sync()

    # --- Pose definitions (right arm) ---
    # L = mirror(R) = -R for symmetric motion
    # J1: forward(+)/backward(-), J2: outward(-)/inward(+), J4: elbow bend(-)
    zero = np.zeros(7)

    # T-pose: arms out to sides (J2 controls outward lift)
    r_tpose = np.array([0.0, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Arms swing forward (J1 controls forward/backward)
    r_forward = np.array([1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Forward + slightly outward (natural reaching pose)
    r_reach = np.array([1.2, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Forward + elbow bent (holding pose)
    r_hold = np.array([1.0, -0.3, 0.0, -1.3, 0.0, 0.0, 0.0])

    # Lift pose (more forward + elbow)
    r_lift = np.array([1.5, -0.3, 0.0, -1.0, 0.0, 0.3, 0.0])

    # Wave pose (arm out + elbow bent + wrist flick)
    r_wave_base = np.array([0.5, -1.5, 0.0, -1.3, 0.0, 0.0, 0.0])
    r_wave_a = r_wave_base.copy()
    r_wave_a[4] = 1.0
    r_wave_b = r_wave_base.copy()
    r_wave_b[4] = -1.0

    print("按 Enter 开始演示...")
    input()

    # ===== Scene 1: T-pose =====
    print("Scene 1: 双臂展开 (T-pose)...")
    backend.movej(r_tpose, mirror(r_tpose), duration=2.0)
    backend.hold(duration=1.0)
    backend.movej(zero, zero, duration=1.5)
    backend.hold(duration=0.5)

    # ===== Scene 2: 双臂前摆 =====
    print("Scene 2: 双臂前摆...")
    backend.movej(r_forward, mirror(r_forward), duration=1.5)
    backend.hold(duration=0.8)
    backend.movej(zero, zero, duration=1.5)
    backend.hold(duration=0.5)

    # ===== Scene 3: 协调搬运 =====
    print("Scene 3: 双臂协调 — 模拟搬运...")
    # Reach out
    backend.movej(r_reach, mirror(r_reach), duration=1.5)
    backend.hold(duration=0.3)
    # Grip (bend elbows)
    backend.movej(r_hold, mirror(r_hold), duration=1.0)
    backend.hold(duration=0.5)
    # Lift
    backend.movej(r_lift, mirror(r_lift), duration=1.2)
    backend.hold(duration=0.8)
    # Put down
    backend.movej(r_hold, mirror(r_hold), duration=1.0)
    backend.hold(duration=0.3)
    # Release
    backend.movej(r_reach, mirror(r_reach), duration=0.8)
    backend.hold(duration=0.3)
    # Return
    backend.movej(zero, zero, duration=1.5)
    backend.hold(duration=0.5)

    # ===== Scene 4: 右臂挥手 =====
    print("Scene 4: 右臂挥手...")
    backend.movej(r_wave_base, zero, duration=1.2)
    backend.hold(duration=0.3)
    for _ in range(4):
        backend.movej(r_wave_a, zero, duration=0.3)
        backend.movej(r_wave_b, zero, duration=0.3)
    backend.hold(duration=0.3)
    backend.movej(zero, zero, duration=1.5)
    backend.hold(duration=1.0)

    print()
    print("=" * 60)
    print("  演示完成!")
    print("=" * 60)

    input("\n按 Enter 关闭...")
    try:
        viewer.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
