"""
MuJoCo 动力学参数辨识演示。

运行方式:
    python examples/15_dynamics_calibration.py

会弹出 MuJoCo viewer 窗口，可以观察标定过程中的运动范围。
"""
import sys
import time
sys.path.insert(0, '.')

from onero_sdk import Arm
from onero_sdk.mujoco_backend import MujocoBackend
import mujoco.viewer
import numpy as np


def main():
    # 创建仿真后端 + viewer
    backend = MujocoBackend(model_name='A1', realtime=True)
    viewer = mujoco.viewer.launch_passive(backend._model, backend._data)
    backend._on_step = lambda: viewer.sync()

    arm = Arm('A1', backend=backend)
    arm.enable()
    viewer.sync()

    print("=" * 60)
    print("  动力学参数辨识 (MuJoCo 仿真演示)")
    print("=" * 60)
    print()
    print("Phase 1: 静态构型采集 — 观察臂在工作空间内的覆盖范围")
    print("Phase 2: 关节摩擦扫描 — 每个关节依次正反转")
    print()
    print("注: 'target clamped' 提示可忽略 — 标定记录的是实际到达位置，")
    print("    不影响辨识精度。")
    print()
    input("按 Enter 开始标定...")

    t0 = time.monotonic()
    result = arm.calibrate_dynamics(
        num_configs=40,
        settle_time=0.5,
        friction_speed=0.5,
        save_path="/tmp/calibration_demo.json",
        seed=42,
    )
    elapsed = time.monotonic() - t0

    print()
    print("=" * 60)
    print(f"  标定完成!")
    print(f"  耗时: {elapsed:.1f}s")
    print(f"  辨识参数维度: {result.rank}")
    print(f"  RMS 残差: {result.residual_rms:.4f} Nm")
    if result.friction_coulomb is not None:
        print(f"  Coulomb 摩擦: {np.round(result.friction_coulomb, 4)}")
        print(f"  粘性摩擦:     {np.round(result.friction_viscous, 4)}")
    print(f"  (MuJoCo 无摩擦模型，故摩擦系数应≈0)")
    print(f"  保存至: /tmp/calibration_demo.json")
    print("=" * 60)

    input("\n按 Enter 关闭...")
    viewer.close()
    arm.disable()


if __name__ == "__main__":
    main()
