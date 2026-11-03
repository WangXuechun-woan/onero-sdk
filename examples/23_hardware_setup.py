"""
CAN 硬件设置 — 使用 SDK 前的硬件准备。

在使用真机之前，需要配置 Linux CAN 接口。
本文件不运行 — 仅作为参考文档。

运行:
    cat examples/23_hardware_setup.py
"""

# =================================================================
# 第 1 步: 配置 CAN 接口 (需要 sudo)
# =================================================================
#
# USB-CAN 适配器 (如 canable, peak-usb):
#   sudo ip link set can0 type can bitrate 1000000
#   sudo ip link set up can0
#
# 验证 CAN 接口存在:
#   ip link show can0
#
# 验证通信 (看到响应帧表示连接正常):
#   candump can0 &
#   cansend can0 001#7F7F7F7F7F7F7F7F
#
# 如果没有 CAN 接口:
#   - 检查 USB 适配器是否插入
#   - dmesg | grep can 查看驱动日志
#   - 可能需要安装驱动: sudo modprobe peak_usb 或 sudo modprobe gs_usb

# =================================================================
# 第 2 步: 安装 SDK
# =================================================================
#
# 基础安装 (仅 CAN 通信):
#   pip install onero-sdk
#
# 带仿真 (MuJoCo):
#   pip install onero-sdk[sim]
#
# 带 IK (逆运动学):
#   pip install onero-sdk[ik]
#
# 带动力学 (重力补偿):
#   pip install onero-sdk[dynamics]
#
# 全部功能:
#   pip install onero-sdk[full]

# =================================================================
# 第 3 步: 最小代码验证
# =================================================================

from onero_sdk import Arm

# 真机连接
arm = Arm("A1R", channel="can0")
arm.enable()

print(f"关节数: {arm.num_joints}")
print(f"当前角度: {arm.state.joints}")

arm.home(duration=3.0)
print("回零完成!")

arm.disable()

# =================================================================
# 常见问题
# =================================================================
#
# Q: "OneroConnectionError: Failed to open can0"
# A: CAN 接口未启动。运行: sudo ip link set up can0
#
# Q: "OneroHardwareError: Failed to enable arm: joint 0: no response"
# A: 硬件未通电 或 CAN 线未连接。检查接线和电源。
#
# Q: 使能后机械臂不动
# A: 可能在关节限位边缘。尝试 arm.home() 回零位。
#
# Q: 想不连硬件测试代码？
# A: 使用 MuJoCo 仿真:
#      from onero_sdk.mujoco_backend import MujocoBackend
#      backend = MujocoBackend(model="A1R")
#      arm = Arm("A1R", backend=backend)
