"""
LeRobot 跨臂遥操作 — 任务空间映射 (末端跟随)。

用 SO101 的末端位移/旋转控制 Onero A1 的末端运动:
  - 位移: 带缩放系数 (补偿臂长差异和运动空间差异)
  - 旋转: 直接传递 (无需缩放)
  - 坐标系: 可配置轴映射 (SO101 坐标系 → A1 坐标系)

核心思路:
  不是映射"绝对关节角"，而是映射"末端的增量运动"。
  每个控制周期计算 SO101 末端相对于初始位姿的位移 Δp 和旋转 ΔR，
  将 Δp 做缩放和坐标变换后，加到 A1 的初始末端位姿上，
  再通过 A1 的逆运动学求解关节角。

优势:
  - 完全解耦臂形: 不要求关节数/连杆长度/安装方式相同
  - 直觉操作: 操作员手往前推，A1 末端也往前走
  - 灵活缩放: 适应不同臂长比例

前提:
    pip install -e lerobot/lerobot_robot_onero
    pip install lerobot numpy

运行:
    python examples/lerobot/05_cross_arm_teleop.py
"""

import time
import math
import numpy as np

from lerobot_robot_onero import OneroRobot, OneroRobotConfig


# ============================================================
# 配置
# ============================================================

# --- 位移缩放 ---
# SO101 臂展约 25cm, A1 臂展约 55cm → 比例 ≈ 2.2
# 即 SO101 末端移动 1cm, A1 末端移动 2.2cm
POSITION_SCALE = 2.2

# 也可以各轴独立缩放 (适应不同方向的运动空间差异)
POSITION_SCALE_XYZ = np.array([2.2, 2.2, 2.0])  # [x, y, z]

# --- 坐标系对齐 ---
# SO101 和 A1 的基座坐标系可能不一致, 需要一个旋转矩阵
# 例: SO101 的 +X 是前方, A1 的 +X 也是前方 → 单位阵
# 例: SO101 的 +X 是前方, A1 的 +Y 是前方 → 绕 Z 旋转 90°
# 这里用轴映射表示: SO101的(x,y,z) → A1的(ax, ay, az)
AXIS_MAP = {
    'x': ('+x', ),   # SO101 +x → A1 +x
    'y': ('+y', ),   # SO101 +y → A1 +y
    'z': ('+z', ),   # SO101 +z → A1 +z
}
# 如果 SO101 的 +x 对应 A1 的 -y: AXIS_MAP = {'x': '-y', 'y': '+x', 'z': '+z'}


def build_axis_rotation(axis_map: dict) -> np.ndarray:
    """
    从轴映射配置构造 3x3 旋转矩阵。

    axis_map: {'x': '+y', 'y': '-x', 'z': '+z'}
    表示 SO101 的 x 轴对应 A1 的 +y 方向, 以此类推。
    """
    R = np.zeros((3, 3))
    axes = {'x': 0, 'y': 1, 'z': 2}

    for src_axis, (mapping,) in axis_map.items():
        sign = 1.0 if mapping[0] == '+' else -1.0
        dst_axis = mapping[1]
        R[axes[dst_axis], axes[src_axis]] = sign

    return R


# --- 运动限制 ---
# A1 末端工作空间边界 (防止超出可达范围)
WORKSPACE_MIN = np.array([0.05, -0.40, 0.02])   # [x_min, y_min, z_min] (m)
WORKSPACE_MAX = np.array([0.60, 0.40, 0.60])    # [x_max, y_max, z_max] (m)

# --- 控制参数 ---
FPS = 50
DEADZONE = 0.002  # 2mm 死区, 消除抖动


# ============================================================
# SO101 正运动学 (简化版)
# ============================================================

class SO101FK:
    """
    SO101 正运动学。
    计算 SO101 末端位姿 (位置 + 姿态)。

    SO101 DH 参数 (近似值, 实际以 URDF 为准):
      link1: d=0.045, a=0,     alpha=pi/2
      link2: d=0,     a=0.104, alpha=0
      link3: d=0,     a=0.088, alpha=0
      link4: d=0,     a=0,     alpha=pi/2
      link5: d=0.055, a=0,     alpha=0
    """

    # 连杆长度 (m)
    L1 = 0.045   # base 到 shoulder
    L2 = 0.104   # upper arm
    L3 = 0.088   # forearm
    L4 = 0.055   # wrist 到 末端

    @staticmethod
    def forward(joints: list[float]) -> tuple[np.ndarray, np.ndarray]:
        """
        计算末端位置和姿态。

        Args:
            joints: 6 个关节角 (弧度)

        Returns:
            (position[3], rotation_matrix[3x3])
        """
        q = joints
        c = [math.cos(q[i]) for i in range(len(q))]
        s = [math.sin(q[i]) for i in range(len(q))]

        # 简化的前向运动学 (5-DOF 位置)
        # 实际项目中应使用完整 DH 或 URDF
        x = (SO101FK.L2 * c[1] + SO101FK.L3 * c[1+1]) * c[0]
        y = (SO101FK.L2 * c[1] + SO101FK.L3 * c[1+1]) * s[0]
        z = SO101FK.L1 + SO101FK.L2 * s[1] + SO101FK.L3 * s[2] + SO101FK.L4

        pos = np.array([x, y, z])

        # 简化的姿态 (末端朝向)
        # 实际应从完整变换矩阵提取
        R = np.eye(3)
        # 绕 Z 旋转 q[0], 绕 Y 旋转 q[1]+q[2]+q[3]
        pitch = q[1] + q[2] + q[3] if len(q) > 3 else q[1] + q[2]
        yaw = q[0]
        roll = q[4] if len(q) > 4 else 0.0

        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)

        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr],
        ])

        return pos, R


# ============================================================
# SO101 Leader 读取
# ============================================================

class SO101Leader:
    """读取 SO101 关节角度, 计算末端位姿。"""

    def __init__(self, port: str = "/dev/ttyACM0"):
        self.port = port
        self._bus = None
        self._fk = SO101FK()
        self._initial_pos = None
        self._initial_rot = None

    def connect(self):
        try:
            from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
            self._bus = FeetechMotorsBus(port=self.port, motors={
                "base": (1, "sts3215"),
                "shoulder": (2, "sts3215"),
                "elbow": (3, "sts3215"),
                "wrist_pitch": (4, "sts3215"),
                "wrist_roll": (5, "sts3215"),
                "gripper": (6, "sts3215"),
            })
            self._bus.connect()
            self._bus.write("Torque_Enable", [0] * 6)
            print(f"SO101 已连接: {self.port}")
        except ImportError:
            print("错误: 需要 lerobot 的 feetech 驱动")
            print("  pip install lerobot")
            raise

    def calibrate_origin(self):
        """记录当前位姿作为初始参考点。"""
        joints = self._read_joints()
        self._initial_pos, self._initial_rot = self._fk.forward(joints)
        print(f"  初始末端位置: [{self._initial_pos[0]:.3f}, "
              f"{self._initial_pos[1]:.3f}, {self._initial_pos[2]:.3f}]")

    def get_delta_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """
        获取相对于初始位姿的末端位移和旋转。

        Returns:
            (delta_position[3], delta_rotation[3x3])
            delta_position: 位移向量 (m)
            delta_rotation: 相对旋转矩阵 (R_current * R_initial^T)
        """
        joints = self._read_joints()
        pos, rot = self._fk.forward(joints)

        delta_pos = pos - self._initial_pos
        delta_rot = rot @ self._initial_rot.T

        return delta_pos, delta_rot

    def _read_joints(self) -> list[float]:
        """读取关节角 (弧度)。"""
        raw = self._bus.read("Present_Position")
        joints = []
        for val in raw:
            angle_rad = (val - 2048) / 2048.0 * math.pi
            joints.append(angle_rad)
        return joints

    def disconnect(self):
        if self._bus:
            self._bus.disconnect()


# ============================================================
# A1 逆运动学接口
# ============================================================

class A1TaskSpaceController:
    """
    A1 任务空间控制: 目标末端位姿 → 关节角。

    内部使用 SDK 的 movel 或阻尼最小二乘逆运动学。
    """

    def __init__(self, robot: OneroRobot):
        self._robot = robot
        self._initial_ee_pos = None
        self._initial_ee_rot = None

    def calibrate_origin(self):
        """记录 A1 当前末端位姿作为初始参考点。"""
        from onero_sdk import Arm
        arm: Arm = self._robot._arm

        # 通过 SDK 获取当前末端位姿
        state = arm.state
        # state.tcp_pose = [x, y, z, rx, ry, rz] (位置 + 轴角)
        self._initial_ee_pos = np.array(state.tcp_pose[:3])
        axis_angle = np.array(state.tcp_pose[3:6])
        self._initial_ee_rot = self._axis_angle_to_matrix(axis_angle)

        print(f"  A1 初始末端位置: [{self._initial_ee_pos[0]:.3f}, "
              f"{self._initial_ee_pos[1]:.3f}, {self._initial_ee_pos[2]:.3f}]")

    def send_target_pose(self, delta_pos: np.ndarray, delta_rot: np.ndarray,
                         scale_xyz: np.ndarray, coord_rotation: np.ndarray):
        """
        发送目标末端位姿。

        Args:
            delta_pos: SO101 末端位移 (已在 SO101 坐标系)
            delta_rot: SO101 末端旋转
            scale_xyz: 位移缩放系数 [sx, sy, sz]
            coord_rotation: 坐标系变换矩阵 (SO101→A1)
        """
        # 1. 位移: 缩放 + 坐标变换
        scaled_delta = delta_pos * scale_xyz
        a1_delta_pos = coord_rotation @ scaled_delta

        # 2. 死区
        if np.linalg.norm(a1_delta_pos) < DEADZONE:
            a1_delta_pos = np.zeros(3)

        # 3. 计算 A1 目标末端位置
        target_pos = self._initial_ee_pos + a1_delta_pos

        # 4. 工作空间限幅
        target_pos = np.clip(target_pos, WORKSPACE_MIN, WORKSPACE_MAX)

        # 5. 旋转: 坐标变换后直接应用 (不缩放)
        a1_delta_rot = coord_rotation @ delta_rot @ coord_rotation.T
        target_rot = a1_delta_rot @ self._initial_ee_rot

        # 6. 转回轴角表示
        target_axis_angle = self._matrix_to_axis_angle(target_rot)

        # 7. 通过 SDK 的 servol 发送末端位姿
        target_pose = list(target_pos) + list(target_axis_angle)
        from onero_sdk import Arm
        arm: Arm = self._robot._arm
        arm.servol(target_pose)

    @staticmethod
    def _axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
        """轴角 → 旋转矩阵 (Rodrigues)。"""
        angle = np.linalg.norm(axis_angle)
        if angle < 1e-10:
            return np.eye(3)
        axis = axis_angle / angle
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ])
        return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K

    @staticmethod
    def _matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
        """旋转矩阵 → 轴角。"""
        angle = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
        if angle < 1e-10:
            return np.zeros(3)
        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ]) / (2 * math.sin(angle))
        return axis * angle


# ============================================================
# 主循环
# ============================================================

def main():
    print("=" * 55)
    print(" 跨臂遥操作: SO101 → Onero A1 (任务空间映射)")
    print("=" * 55)
    print()
    print("原理:")
    print("  SO101 末端位移 × 缩放 × 坐标变换 → A1 末端位移")
    print("  SO101 末端旋转 (无缩放) → A1 末端旋转")
    print()
    print(f"配置:")
    print(f"  位移缩放: {POSITION_SCALE_XYZ.tolist()}")
    print(f"  工作空间: x[{WORKSPACE_MIN[0]:.2f}, {WORKSPACE_MAX[0]:.2f}] "
          f"y[{WORKSPACE_MIN[1]:.2f}, {WORKSPACE_MAX[1]:.2f}] "
          f"z[{WORKSPACE_MIN[2]:.2f}, {WORKSPACE_MAX[2]:.2f}]")
    print()

    # 坐标系变换矩阵
    coord_R = build_axis_rotation(AXIS_MAP)

    # 连接 follower (A1)
    print("连接 A1 (follower)...")
    robot = OneroRobot(OneroRobotConfig(can_channel="can0"))
    robot.connect()

    # 连接 leader (SO101)
    print("连接 SO101 (leader)...")
    leader = SO101Leader(port="/dev/ttyACM0")
    leader.connect()

    # 任务空间控制器
    controller = A1TaskSpaceController(robot)

    # 校准初始位姿
    print("\n校准初始位姿...")
    print("  请将两台臂放到舒适的操作位, 然后按 Enter")
    input()
    leader.calibrate_origin()
    controller.calibrate_origin()

    dt = 1.0 / FPS

    print(f"\n遥操作启动! ({FPS} Hz)")
    print("移动 SO101 末端 → A1 末端同步跟随")
    print("按 Ctrl+C 停止\n")

    frame_count = 0
    try:
        while True:
            t0 = time.monotonic()

            # 1. 获取 SO101 末端增量
            delta_pos, delta_rot = leader.get_delta_pose()

            # 2. 映射到 A1 并执行
            controller.send_target_pose(
                delta_pos, delta_rot,
                scale_xyz=POSITION_SCALE_XYZ,
                coord_rotation=coord_R,
            )

            # 3. 状态打印
            frame_count += 1
            if frame_count % FPS == 0:
                print(f"  SO101 Δpos: [{delta_pos[0]:+.3f}, "
                      f"{delta_pos[1]:+.3f}, {delta_pos[2]:+.3f}] m")

            # 帧率控制
            elapsed = time.monotonic() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\n停止遥操作")

    finally:
        leader.disconnect()
        robot.disconnect()
        print("已断开连接")


if __name__ == "__main__":
    main()
