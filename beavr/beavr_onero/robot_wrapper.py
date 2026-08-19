"""
BEAVR-Bot Onero A1 适配器 — RobotWrapper 实现。

将 Onero A1 机械臂接入 BEAVR VR 遥操作框架。
BEAVR 通过 ZeroMQ 发布 VR 手柄的 6-DOF 位姿，
本适配器接收目标位姿并通过 SDK 的 servol 执行。

架构:
    Meta Quest 3 → VR Detector → Transform → Operator → [本文件] → A1 硬件

使用:
    # 在 BEAVR 的 teleop config 中注册:
    python -m beavr.teleop.run --robot=onero_a1 --laterality=right

参考:
    https://github.com/ARCLab-MIT/beavr-bot/
"""

from __future__ import annotations

import logging
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Any

from onero_sdk import Arm
from onero_sdk.protocols import ArmProtocol
from onero_sdk.fk import URDFChain
from onero_sdk.model_registry import _find_urdf
from onero_sdk.safety import SafetyConfig

logger = logging.getLogger(__name__)


# ============================================================
# 坐标系变换: VR (Y-up 右手系) → A1 (Z-up 右手系)
# ============================================================

# VR 标准: X=右, Y=上, Z=后 (朝向用户)
# A1 基座: X=前, Y=左, Z=上
# 变换: A1_x = -VR_z, A1_y = -VR_x, A1_z = VR_y
R_VR_TO_A1 = np.array([
    [0.0, 0.0, -1.0],   # A1_x = -VR_z
    [-1.0, 0.0, 0.0],   # A1_y = -VR_x
    [0.0, 1.0, 0.0],    # A1_z =  VR_y
], dtype=np.float64)


# ============================================================
# 配置
# ============================================================

@dataclass
class OneroA1Config:
    """Onero A1 BEAVR 适配器配置。"""

    # 硬件
    can_channel: str = "can0"
    model: str = "A1R"
    mounting: Any = "side_mount_left"
    world_frame: bool = False
    safety: Optional[SafetyConfig] = None

    # 位移缩放 (VR 空间 → A1 工作空间)
    # Meta Quest 手柄活动范围约 ±0.5m, A1 工作空间约 ±0.4m
    position_scale: float = 0.8
    position_scale_xyz: list[float] = field(
        default_factory=lambda: [0.8, 0.8, 0.7])

    # 坐标系旋转矩阵 (VR → A1)
    # 默认: 标准 VR Y-up → A1 Z-up
    # 如果 VR 朝向和机械臂朝向不同, 可以额外叠加一个 Yaw 旋转
    yaw_offset_deg: float = 0.0  # VR 前方 vs A1 前方的偏转角

    # 工作空间限制 (m, A1 基座坐标系)
    workspace_min: list[float] = field(
        default_factory=lambda: [0.05, -0.40, 0.02])
    workspace_max: list[float] = field(
        default_factory=lambda: [0.60, 0.40, 0.60])

    # 控制参数
    control_freq: float = 60.0   # 控制频率 (Hz)
    smoothing_alpha: float = 0.3  # 低通滤波系数 (0=全滤波, 1=无滤波)
    adaptive_smoothing: bool = True
    min_smoothing_alpha: float = 0.15
    max_smoothing_alpha: float = 0.85
    deadzone: float = 0.003       # 3mm 死区
    command_frame: str = "base"

    # 安全
    max_linear_speed: float = 0.5    # m/s 最大末端速度
    max_angular_speed: float = 2.0   # rad/s 最大角速度
    max_position_jump: float = 0.50   # m 单帧硬跳变阈值
    max_rotation_jump: float = 1.2    # rad 单帧硬跳变阈值


# ============================================================
# RobotWrapper 实现
# ============================================================

class OneroA1Robot:
    """
    BEAVR RobotWrapper 实现 — Onero A1 机械臂。

    接口方法 (BEAVR RobotWrapper API):
      - connect()       连接硬件
      - disconnect()    断开连接
      - get_state()     获取当前状态
      - send_action()   发送控制指令
      - reset()         复位到初始位
    """

    def __init__(self, config: Optional[OneroA1Config] = None):
        self.config = config or OneroA1Config()
        self._arm: Optional[ArmProtocol] = None
        self._fk: Optional[URDFChain] = None
        self._connected = False

        # 状态缓存
        self._current_ee_pos = np.zeros(3)
        self._current_ee_rot = np.eye(3)
        self._current_joints = np.zeros(7)

        # 初始参考位姿 (VR 校准时记录)
        self._vr_origin_pos = None
        self._vr_origin_rot = None
        self._arm_origin_pos = None
        self._arm_origin_rot = None

        # 低通滤波状态
        self._filtered_pos = None
        self._filtered_rot = None
        self._last_command_time = None
        self._last_raw_target_pos = None
        self._last_raw_target_rot = None

        # 坐标系变换
        self._R_vr_to_arm = self._build_rotation_matrix()

    # ----------------------------------------------------------
    # BEAVR RobotWrapper API
    # ----------------------------------------------------------

    def connect(self):
        """连接机械臂硬件。"""
        self._arm = Arm(
            self.config.model,
            channel=self.config.can_channel,
            mounting=self.config.mounting,
            world_frame=self.config.world_frame,
            safety=self.config.safety,
        )
        self._arm.enable()

        # 初始化 FK
        urdf_path = _find_urdf(self.config.model)
        self._fk = URDFChain(str(urdf_path))

        # 读取初始状态
        self._update_state()
        self._arm_origin_pos = self._current_ee_pos.copy()
        self._arm_origin_rot = self._current_ee_rot.copy()
        self._filtered_pos = self._current_ee_pos.copy()
        self._filtered_rot = self._current_ee_rot.copy()
        self._last_command_time = time.monotonic()
        self._last_raw_target_pos = self._current_ee_pos.copy()
        self._last_raw_target_rot = self._current_ee_rot.copy()

        self._connected = True
        print(f"[OneroA1Robot] 已连接: {self.config.can_channel}, "
              f"末端初始位置: [{self._arm_origin_pos[0]:.3f}, "
              f"{self._arm_origin_pos[1]:.3f}, {self._arm_origin_pos[2]:.3f}]")

    def disconnect(self):
        """断开连接。"""
        if self._arm:
            try:
                self._arm.disable()
            except Exception:
                pass
        self._connected = False
        print("[OneroA1Robot] 已断开")

    def get_state(self) -> dict:
        """
        获取当前机械臂状态。

        Returns:
            {
                "joint_positions": np.array[7],
                "joint_velocities": np.array[7],
                "ee_position": np.array[3],     # 末端位置 (m)
                "ee_rotation": np.array[3x3],   # 末端旋转矩阵
                "ee_pose_6d": np.array[6],      # [x, y, z, rx, ry, rz]
                "timestamp": float,
            }
        """
        self._update_state()
        state = self._arm.state

        return {
            "joint_positions": np.array(state.joints),
            "joint_velocities": np.array(state.velocities),
            "ee_position": self._current_ee_pos.copy(),
            "ee_rotation": self._current_ee_rot.copy(),
            "ee_pose_6d": np.array(state.tcp_pose) if hasattr(state, 'tcp_pose') else None,
            "timestamp": time.time(),
        }

    def send_action(self, action: dict):
        """
        发送控制指令。

        Args:
            action: {
                "target_position": np.array[3],  # VR 坐标系下的目标位置
                "target_rotation": np.array[3x3], # VR 坐标系下的目标旋转
            }
            或 BEAVR Operator 输出格式:
            {
                "ee_pose": np.array[4x4],  # 4x4 齐次变换 (VR 坐标系)
            }
        """
        if not self._connected:
            return

        # 解析 action
        if "ee_pose" in action:
            # BEAVR 标准格式: 4x4 齐次变换矩阵
            T = np.array(action["ee_pose"])
            if T.shape != (4, 4) or not np.all(np.isfinite(T)):
                logger.warning("Ignoring invalid ee_pose action")
                return
            vr_pos = T[:3, 3]
            vr_rot = self._project_to_rotation_matrix(T[:3, :3])
        elif "target_position" in action:
            vr_pos = np.array(action["target_position"])
            if vr_pos.shape != (3,) or not np.all(np.isfinite(vr_pos)):
                logger.warning("Ignoring invalid target_position action")
                return
            vr_rot = self._project_to_rotation_matrix(np.array(action.get("target_rotation", np.eye(3))))
        else:
            return

        # 首次接收 VR 数据时记录原点
        if self._vr_origin_pos is None:
            self._vr_origin_pos = vr_pos.copy()
            self._vr_origin_rot = vr_rot.copy()
            print(f"[OneroA1Robot] VR 原点已校准: [{vr_pos[0]:.3f}, "
                  f"{vr_pos[1]:.3f}, {vr_pos[2]:.3f}]")
            return

        # 计算 VR 增量
        delta_pos_vr = vr_pos - self._vr_origin_pos
        delta_rot_vr = vr_rot @ self._vr_origin_rot.T

        # 坐标变换: VR → A1
        scale = np.array(self.config.position_scale_xyz)
        delta_pos_arm = self._R_vr_to_arm @ (delta_pos_vr * scale)
        delta_rot_arm = self._project_to_rotation_matrix(
            self._R_vr_to_arm @ delta_rot_vr @ self._R_vr_to_arm.T
        )

        # 死区
        if np.linalg.norm(delta_pos_arm) < self.config.deadzone:
            delta_pos_arm = np.zeros(3)

        # 计算目标位姿
        target_pos = self._arm_origin_pos + delta_pos_arm
        target_rot = self._project_to_rotation_matrix(delta_rot_arm @ self._arm_origin_rot)

        # 工作空间限幅
        ws_min = np.array(self.config.workspace_min)
        ws_max = np.array(self.config.workspace_max)
        target_pos = np.clip(target_pos, ws_min, ws_max)

        now = time.monotonic()
        dt = self._command_dt(now)

        # 丢弃明显不可信的输入跳变；速度限幅处理正常大动作。
        if self._is_hard_jump(target_pos, target_rot):
            logger.warning("Ignoring implausible VR pose jump")
            return

        # 速度限制
        target_pos = self._apply_speed_limit(target_pos, dt)
        target_rot = self._apply_angular_speed_limit(target_rot, dt)

        # 低通滤波 (消除 VR 手部抖动)
        alpha = self._effective_smoothing_alpha(target_pos, dt)
        self._filtered_pos = alpha * target_pos + (1 - alpha) * self._filtered_pos
        self._filtered_rot = self._slerp_matrix(
            self._filtered_rot, target_rot, alpha)
        self._last_command_time = now
        self._last_raw_target_pos = target_pos.copy()
        self._last_raw_target_rot = target_rot.copy()

        # 转换为轴角并发送
        target_pose_6d = self._to_pose_6d(self._filtered_pos, self._filtered_rot)
        self._arm.servol(target_pose_6d, frame=self.config.command_frame)

    def reset(self):
        """复位到初始位姿。"""
        if self._arm:
            self._arm.movej([0.0, -0.5, 0.0, -1.0, 0.0, 0.5, 0.0], duration=3.0)
            self._update_state()
            self._arm_origin_pos = self._current_ee_pos.copy()
            self._arm_origin_rot = self._current_ee_rot.copy()
            self._filtered_pos = self._current_ee_pos.copy()
            self._filtered_rot = self._current_ee_rot.copy()
            self._vr_origin_pos = None
            self._vr_origin_rot = None
            self._last_command_time = time.monotonic()
            self._last_raw_target_pos = self._current_ee_pos.copy()
            self._last_raw_target_rot = self._current_ee_rot.copy()
            print("[OneroA1Robot] 已复位")

    def calibrate_vr_origin(self):
        """手动触发 VR 原点校准 (下一帧 VR 数据作为原点)。"""
        self._vr_origin_pos = None
        self._vr_origin_rot = None
        self._arm_origin_pos = self._current_ee_pos.copy()
        self._arm_origin_rot = self._current_ee_rot.copy()
        self._filtered_pos = self._current_ee_pos.copy()
        self._filtered_rot = self._current_ee_rot.copy()
        self._last_command_time = time.monotonic()
        self._last_raw_target_pos = self._current_ee_pos.copy()
        self._last_raw_target_rot = self._current_ee_rot.copy()
        print("[OneroA1Robot] VR 原点已清除, 等待重新校准...")

    def hold(self):
        """安全暂停遥操作并保持当前位置。"""
        if not self._arm or not self._connected:
            return
        try:
            if hasattr(self._arm, "stop"):
                self._arm.stop()
            else:
                self._update_state()
                self._arm.servol(
                    self._to_pose_6d(self._current_ee_pos, self._current_ee_rot),
                    frame=self.config.command_frame,
                )
        except Exception as exc:
            logger.warning("Failed to hold Onero A1 arm: %s", exc)

    # ----------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------

    def _update_state(self):
        """从 FK 更新末端位姿。"""
        state = self._arm.state
        self._current_joints = np.array(state.joints)
        pos, rot = self._fk.get_ee_pose(state.joints)
        self._current_ee_pos = np.array(pos)
        self._current_ee_rot = np.array(rot)

    def _build_rotation_matrix(self) -> np.ndarray:
        """构建 VR→A1 坐标变换矩阵 (包含 yaw offset)。"""
        yaw = np.radians(self.config.yaw_offset_deg)
        if abs(yaw) < 1e-6:
            return R_VR_TO_A1.copy()
        # 额外绕 A1 Z 轴旋转
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ])
        return Rz @ R_VR_TO_A1

    def _command_dt(self, now: float) -> float:
        if self._last_command_time is None:
            return 1.0 / self.config.control_freq
        dt = now - self._last_command_time
        nominal = 1.0 / self.config.control_freq
        return float(np.clip(dt, nominal * 0.25, 0.2))

    def _is_hard_jump(self, target_pos: np.ndarray, target_rot: np.ndarray) -> bool:
        if self._last_raw_target_pos is None or self._last_raw_target_rot is None:
            return False
        pos_jump = np.linalg.norm(target_pos - self._last_raw_target_pos)
        rot_jump = self._rotation_angle(self._last_raw_target_rot.T @ target_rot)
        return (
            pos_jump > self.config.max_position_jump
            or rot_jump > self.config.max_rotation_jump
        )

    def _effective_smoothing_alpha(self, target_pos: np.ndarray, dt: float) -> float:
        base = float(np.clip(self.config.smoothing_alpha, 0.0, 1.0))
        if base <= 0.0 or base >= 1.0 or not self.config.adaptive_smoothing:
            return base

        speed = np.linalg.norm(target_pos - self._filtered_pos) / max(dt, 1e-6)
        # Slow hand motion is usually tremor: filter harder. Intentional fast
        # motion should respond with less lag.
        t = np.clip(speed / max(self.config.max_linear_speed, 1e-6), 0.0, 1.0)
        alpha = self.config.min_smoothing_alpha * (1.0 - t) + self.config.max_smoothing_alpha * t
        return float(np.clip(alpha, 0.0, 1.0))

    def _apply_speed_limit(self, target_pos: np.ndarray, dt: float) -> np.ndarray:
        """限制末端线速度。"""
        max_step = self.config.max_linear_speed * dt
        delta = target_pos - self._filtered_pos
        dist = np.linalg.norm(delta)
        if dist > max_step:
            target_pos = self._filtered_pos + delta * (max_step / dist)
        return target_pos

    def _apply_angular_speed_limit(self, target_rot: np.ndarray, dt: float) -> np.ndarray:
        """限制末端角速度。"""
        delta_rot = self._filtered_rot.T @ target_rot
        angle = self._rotation_angle(delta_rot)
        max_step = self.config.max_angular_speed * dt
        if angle <= max_step or angle < 1e-9:
            return target_rot
        limited_delta = self._matrix_power(delta_rot, max_step / angle)
        return self._project_to_rotation_matrix(self._filtered_rot @ limited_delta)

    @staticmethod
    def _slerp_matrix(R_from: np.ndarray, R_to: np.ndarray,
                      alpha: float) -> np.ndarray:
        """简化的旋转矩阵插值 (近似 SLERP)。"""
        # 对于小角度差异, 线性插值 + 正交化足够
        R_interp = (1 - alpha) * R_from + alpha * R_to
        return OneroA1Robot._project_to_rotation_matrix(R_interp)

    @staticmethod
    def _project_to_rotation_matrix(rot: np.ndarray) -> np.ndarray:
        rot = np.asarray(rot, dtype=float)
        if rot.shape != (3, 3) or not np.all(np.isfinite(rot)):
            return np.eye(3)
        u, _, vt = np.linalg.svd(rot)
        fixed = u @ vt
        if np.linalg.det(fixed) < 0:
            vt[-1, :] *= -1.0
            fixed = u @ vt
        return fixed

    @staticmethod
    def _rotation_angle(rot: np.ndarray) -> float:
        rot = OneroA1Robot._project_to_rotation_matrix(rot)
        cos_angle = (np.trace(rot) - 1.0) / 2.0
        return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    @staticmethod
    def _matrix_power(rot: np.ndarray, scale: float) -> np.ndarray:
        angle = OneroA1Robot._rotation_angle(rot)
        if angle < 1e-9:
            return np.eye(3)
        axis = np.array([
            rot[2, 1] - rot[1, 2],
            rot[0, 2] - rot[2, 0],
            rot[1, 0] - rot[0, 1],
        ]) / (2.0 * np.sin(angle))
        return OneroA1Robot._rotvec_to_matrix(axis * angle * scale)

    @staticmethod
    def _rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
        angle = np.linalg.norm(rotvec)
        if angle < 1e-10:
            return np.eye(3)
        axis = rotvec / angle
        x, y, z = axis
        K = np.array([
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ])
        return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)

    @staticmethod
    def _to_pose_6d(pos: np.ndarray, rot: np.ndarray) -> list[float]:
        """位置 + 旋转矩阵 → [x, y, z, rx, ry, rz] 轴角表示。"""
        import math
        angle = math.acos(max(-1.0, min(1.0, (np.trace(rot) - 1) / 2)))
        if angle < 1e-10:
            rvec = [0.0, 0.0, 0.0]
        else:
            axis = np.array([
                rot[2, 1] - rot[1, 2],
                rot[0, 2] - rot[2, 0],
                rot[1, 0] - rot[0, 1],
            ]) / (2 * math.sin(angle))
            rvec = (axis * angle).tolist()
        return pos.tolist() + rvec
