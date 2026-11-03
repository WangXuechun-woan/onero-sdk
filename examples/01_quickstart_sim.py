"""
快速入门 — 使用 MuJoCo 仿真，无需真机。

这是最简单的入门示例，5 行代码就能让机械臂动起来。
无需 CAN 硬件，只需安装仿真依赖:
    pip install onero-sdk[sim]

运行:
    python examples/01_quickstart_sim.py
"""

import time

import mujoco.viewer

from onero_sdk import Arm

try:
    from onero_sdk.mujoco_backend import MujocoBackend
except ImportError:
    raise SystemExit(
        "仿真后端需要安装 MuJoCo:\n"
        "    pip install onero-sdk[sim]\n"
        "然后重新运行此示例。"
    )

# 创建仿真后端（替代真实 CAN 硬件）
backend = MujocoBackend(model_name="A1R", realtime=True)
viewer = mujoco.viewer.launch_passive(backend._model, backend._data)
backend._on_step = viewer.sync

try:
    # 用仿真后端创建 Arm（接口与真机完全一致）
    arm = Arm("A1R", backend=backend)
    arm.enable()

    # 移动到目标关节角度
    arm.movej([0.0, -0.5, 0.3, -1.0, 0.0, 0.5, 0.0], duration=2.0)
    print(f"到达位置: {[f'{j:.2f}' for j in arm.state.joints]}")

    # 回到零位
    arm.home(duration=2.0)
    print("已回零位")

    print("窗口保持打开，关闭 MuJoCo 窗口即可退出。")
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.01)
finally:
    arm.disable()
    backend.close()
    viewer.close()
