"""
BEAVR-Bot Onero A1 Operator — VR 手柄位姿到机械臂指令的转换。

Operator 是 BEAVR 架构中的核心组件:
  VR Detector → Transform → [Operator] → RobotWrapper

本 Operator:
  1. 订阅 ZMQ 发布的稳定 6-DOF 手部位姿
  2. 应用坐标变换 + 位移缩放
  3. 发布动作指令给 RobotWrapper
"""

from __future__ import annotations

import json
import logging
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

try:
    import zmq
except ImportError:
    zmq = None
    print("警告: zmq 未安装, 请 pip install pyzmq")

from .robot_wrapper import OneroA1Robot, OneroA1Config

logger = logging.getLogger(__name__)


@dataclass
class OneroOperatorConfig:
    """Operator 配置。"""

    # ZMQ 端口
    host: str = "localhost"
    vr_sub_port: int = 5555        # 订阅 VR 手部位姿
    state_pub_port: int = 5556     # 发布机械臂状态
    command_pub_port: int = 5557   # 发布指令 (供 recorder 记录)
    zmq_hwm: int = 1
    receive_timeout_ms: int = 5

    # VR 话题
    vr_topic: str = "hand_right"   # 订阅的 ZMQ topic

    # 控制
    vr_freq: float = 90.0          # VR 输入频率
    control_freq: float = 60.0     # 控制输出频率 (servol 频率)
    max_frame_age_s: float = 0.20  # 超过该时长无新 VR 帧则暂停遥操作
    hold_on_stale: bool = True
    hold_on_pause: bool = True

    # 机械臂配置
    robot_config: OneroA1Config = field(default_factory=OneroA1Config)

    # 按钮映射
    trigger_button: str = "trigger"     # 触发遥操作
    grip_button: str = "grip"           # 夹爪控制
    reset_button: str = "menu"          # 复位
    enable_gripper: bool = False


class OneroA1Operator:
    """
    BEAVR Operator — 将 VR 手部位姿转换为 Onero A1 动作。

    使用方式:
        operator = OneroA1Operator(config)
        operator.start()  # 阻塞运行

    或在 BEAVR 框架中:
        注册到 teleop config, 由 BEAVR 主循环驱动
    """

    def __init__(self, config: Optional[OneroOperatorConfig] = None):
        self.config = config or OneroOperatorConfig()
        self._robot = OneroA1Robot(self.config.robot_config)
        self._running = False

        # ZMQ
        self._context = None
        self._vr_subscriber = None
        self._state_publisher = None
        self._command_publisher = None

        # 状态
        self._is_active = False  # trigger 按下 = 激活遥操作
        self._gripper_closed = False
        self._last_vr_rx_time: Optional[float] = None
        self._last_vr_msg_time: Optional[float] = None
        self._stale_hold_sent = False
        self._reset_pressed = False

    def start(self):
        """启动 operator 主循环 (阻塞)。"""
        self._setup_zmq()
        self._robot.connect()
        self._running = True

        dt = 1.0 / self.config.control_freq
        print(f"[OneroA1Operator] 运行中 @ {self.config.control_freq} Hz")
        print(f"  订阅 VR: tcp://{self.config.host}:{self.config.vr_sub_port} "
              f"[topic={self.config.vr_topic}]")
        print(f"  按 VR trigger 键激活遥操作")

        try:
            while self._running:
                t0 = time.monotonic()
                self._step()
                elapsed = time.monotonic() - t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)
        except KeyboardInterrupt:
            print("\n[OneroA1Operator] 停止")
        finally:
            self.cleanup()

    def step(self, vr_data: Optional[dict] = None) -> dict:
        """
        单步执行 (供 BEAVR 框架调用)。

        Args:
            vr_data: 外部传入的 VR 数据 (不从 ZMQ 读取)

        Returns:
            当前机械臂状态
        """
        if vr_data is not None:
            self._process_vr_data(vr_data)
        else:
            self._step()
        return self._robot.get_state()

    def cleanup(self):
        """清理资源。"""
        self._running = False
        self._robot.disconnect()
        if self._vr_subscriber:
            self._vr_subscriber.close()
        if self._state_publisher:
            self._state_publisher.close()
        if self._command_publisher:
            self._command_publisher.close()
        if self._context:
            self._context.term()

    # ----------------------------------------------------------
    # 内部
    # ----------------------------------------------------------

    def _setup_zmq(self):
        """初始化 ZeroMQ 套接字。"""
        if zmq is None:
            raise ImportError("需要 pyzmq: pip install pyzmq")

        self._context = zmq.Context()

        # 订阅 VR 手部位姿
        self._vr_subscriber = self._context.socket(zmq.SUB)
        self._vr_subscriber.setsockopt(zmq.RCVHWM, self.config.zmq_hwm)
        self._vr_subscriber.setsockopt(zmq.LINGER, 0)
        self._vr_subscriber.setsockopt(zmq.RCVTIMEO, self.config.receive_timeout_ms)
        self._vr_subscriber.connect(f"tcp://{self.config.host}:{self.config.vr_sub_port}")
        self._vr_subscriber.setsockopt_string(zmq.SUBSCRIBE, self.config.vr_topic)

        # 发布机械臂状态
        self._state_publisher = self._context.socket(zmq.PUB)
        self._state_publisher.setsockopt(zmq.SNDHWM, self.config.zmq_hwm)
        self._state_publisher.setsockopt(zmq.LINGER, 0)
        self._state_publisher.bind(f"tcp://*:{self.config.state_pub_port}")

        # 发布指令 (供 recorder 记录)
        self._command_publisher = self._context.socket(zmq.PUB)
        self._command_publisher.setsockopt(zmq.SNDHWM, self.config.zmq_hwm)
        self._command_publisher.setsockopt(zmq.LINGER, 0)
        self._command_publisher.bind(f"tcp://*:{self.config.command_pub_port}")

    def _step(self):
        """单个控制周期。"""
        # 接收 VR 数据
        vr_data = self._receive_vr()
        if vr_data is None:
            self._handle_missing_vr_frame()
            return

        self._process_vr_data(vr_data)

        # 发布状态
        state = self._robot.get_state()
        self._publish_state(state)

    def _process_vr_data(self, vr_data: dict):
        """处理 VR 数据并执行动作。"""
        # 检查按钮状态
        buttons = vr_data.get("buttons", {})

        # Trigger: 激活/停用遥操作
        trigger = buttons.get(self.config.trigger_button, False)
        if trigger and not self._is_active:
            self._is_active = True
            self._stale_hold_sent = False
            self._robot.calibrate_vr_origin()
            print("[OneroA1Operator] 遥操作激活!")
        elif not trigger and self._is_active:
            self._is_active = False
            if self.config.hold_on_pause:
                self._robot.hold()
            print("[OneroA1Operator] 遥操作暂停")

        # Reset
        reset_pressed = buttons.get(self.config.reset_button, False)
        if reset_pressed and not self._reset_pressed:
            self._robot.reset()
            self._is_active = False
            self._reset_pressed = True
            return
        if not reset_pressed:
            self._reset_pressed = False

        # 仅在激活时发送动作
        if not self._is_active:
            return

        # 构造 action
        action = {}
        if "pose_matrix" in vr_data:
            action["ee_pose"] = np.array(vr_data["pose_matrix"]).reshape(4, 4)
        elif "position" in vr_data and "rotation" in vr_data:
            T = np.eye(4)
            T[:3, 3] = np.array(vr_data["position"])
            T[:3, :3] = np.array(vr_data["rotation"]).reshape(3, 3)
            action["ee_pose"] = T
        else:
            return

        # 发送到机械臂
        self._robot.send_action(action)

        # 发布指令 (供数据采集记录)
        self._publish_command(action)

        # 夹爪控制
        grip = buttons.get(self.config.grip_button, False)
        if grip != self._gripper_closed:
            self._gripper_closed = grip
            if self.config.enable_gripper:
                logger.warning("Gripper control requested but no Onero gripper interface is configured")
            else:
                logger.debug("Ignoring gripper button because gripper control is disabled")

    def _receive_vr(self) -> Optional[dict]:
        """从 ZMQ 接收 VR 手部数据。"""
        try:
            # 多帧积压时只取最新
            msg = None
            while True:
                try:
                    msg = self._vr_subscriber.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break

            if msg is None:
                # 阻塞等待一帧
                try:
                    msg = self._vr_subscriber.recv_multipart()
                except zmq.Again:
                    return None

            if len(msg) < 2:
                return None

            data = json.loads(msg[1].decode())
            now = time.monotonic()
            self._last_vr_rx_time = now
            self._last_vr_msg_time = self._extract_message_time(data)
            if self._is_frame_stale(now):
                logger.warning("Dropping stale VR frame")
                return None
            return data

        except Exception as exc:
            logger.debug("Failed to receive VR frame: %s", exc)
            return None

    def _extract_message_time(self, data: dict) -> Optional[float]:
        for key in ("timestamp_s", "timestamp", "time_s", "t"):
            value = data.get(key)
            if isinstance(value, (int, float)) and np.isfinite(value):
                # Wall-clock timestamps are converted to this process' monotonic
                # clock for age checks. Small timestamps are treated as local
                # monotonic-like values and only used if comparable.
                if value > 1_000_000:
                    return time.monotonic() - max(0.0, time.time() - float(value))
                return None
        return None

    def _is_frame_stale(self, now: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else now
        if self._last_vr_rx_time is None:
            return False
        if now - self._last_vr_rx_time > self.config.max_frame_age_s:
            return True
        if self._last_vr_msg_time is not None:
            msg_age = now - self._last_vr_msg_time
            if 0 <= msg_age > self.config.max_frame_age_s:
                return True
        return False

    def _handle_missing_vr_frame(self):
        if not self._is_active or not self._is_frame_stale():
            return
        self._is_active = False
        if self.config.hold_on_stale and not self._stale_hold_sent:
            self._robot.hold()
            self._stale_hold_sent = True
        logger.warning("VR stream stale; teleoperation paused until trigger is re-engaged")

    def _publish_state(self, state: dict):
        """发布机械臂状态到 ZMQ。"""
        if self._state_publisher is None:
            return
        # 序列化 numpy
        serializable = {
            "joint_positions": state["joint_positions"].tolist(),
            "joint_velocities": state["joint_velocities"].tolist(),
            "ee_position": state["ee_position"].tolist(),
            "timestamp": state["timestamp"],
        }
        try:
            self._state_publisher.send_multipart([
                b"onero_state",
                json.dumps(serializable).encode(),
            ], flags=zmq.NOBLOCK)
        except zmq.Again:
            pass

    def _publish_command(self, action: dict):
        """发布指令到 ZMQ (供 recorder 记录)。"""
        if self._command_publisher is None:
            return
        serializable = {}
        if "ee_pose" in action:
            serializable["ee_pose"] = action["ee_pose"].tolist()
        try:
            self._command_publisher.send_multipart([
                b"onero_command",
                json.dumps(serializable).encode(),
            ], flags=zmq.NOBLOCK)
        except zmq.Again:
            pass
