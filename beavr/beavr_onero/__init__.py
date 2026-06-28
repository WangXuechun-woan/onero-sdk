"""
BEAVR Onero A1 适配器包。

提供:
  - OneroA1Robot: RobotWrapper 实现 (接收指令, 控制硬件)
  - OneroA1Operator: Operator 实现 (VR→机械臂指令转换)
  - OneroA1Config: 配置类
  - MountingPreset: 安装方式预设

使用:
    from beavr_onero import OneroA1Robot, OneroA1Config, MountingPreset

    config = OneroA1Config(
        can_channel="can0",
        mounting=MountingPreset.SIDE_MOUNT_LEFT,  # 侧装 (装在支架上)
    )
    robot = OneroA1Robot(config)
    robot.connect()
"""

from .robot_wrapper import OneroA1Robot, OneroA1Config, R_VR_TO_A1
from .operator import OneroA1Operator, OneroOperatorConfig
from onero_sdk.mounting import MountingPreset, get_mounting_rotation

__all__ = [
    "OneroA1Robot",
    "OneroA1Config",
    "OneroA1Operator",
    "OneroOperatorConfig",
    "MountingPreset",
    "get_mounting_rotation",
]
