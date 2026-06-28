"""
MoveIt2 笛卡尔规划 — 直线和圆弧路径。

使用 MoveIt2 的笛卡尔路径规划 (Cartesian path planning) 控制末端执行器
沿直线或圆弧运动。适合焊接、涂胶等需要精确路径的任务。

前提:
    ros2 launch onero_moveit_config move_group.launch.py

运行:
    python examples/moveit/02_moveit_cartesian.py
"""

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point, Quaternion
from moveit.planning import MoveItPy
from moveit.core.robot_state import RobotState


def make_pose(x, y, z, qx=0.0, qy=0.707, qz=0.0, qw=0.707) -> Pose:
    """创建 Pose 消息 (默认末端朝下)。"""
    pose = Pose()
    pose.position = Point(x=x, y=y, z=z)
    pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
    return pose


def main():
    rclpy.init()

    print("初始化 MoveIt...")
    moveit = MoveItPy(node_name="moveit_cartesian_example")
    arm = moveit.get_planning_component("arm")

    print("MoveIt 就绪!\n")

    # === 1. 直线运动 ===
    print("--- 1. 笛卡尔直线运动 ---")
    print("  末端在 XY 平面画一个正方形")

    # 先到起始位
    arm.set_start_state_to_current_state()
    arm.set_goal_state(configuration_name="home")
    result = arm.plan()
    if result:
        arm.execute()

    # 定义正方形路径点
    side = 0.1  # 10cm 边长
    # 假设 home 位姿下末端在 (0.3, 0, 0.4) 附近
    cx, cy, cz = 0.3, 0.0, 0.35
    square_waypoints = [
        make_pose(cx + side/2, cy + side/2, cz),
        make_pose(cx - side/2, cy + side/2, cz),
        make_pose(cx - side/2, cy - side/2, cz),
        make_pose(cx + side/2, cy - side/2, cz),
        make_pose(cx + side/2, cy + side/2, cz),  # 回到起点
    ]

    arm.set_start_state_to_current_state()

    # 笛卡尔路径规划
    robot_model = moveit.get_robot_model()
    robot_state = RobotState(robot_model)
    robot_state.update()  # 从当前状态更新

    fraction = arm.compute_cartesian_path(
        waypoints=square_waypoints,
        max_step=0.005,  # 5mm 插值步长
        jump_threshold=0.0,  # 禁止关节跳变
    )

    if fraction > 0.9:
        print(f"  规划覆盖率: {fraction*100:.1f}%")
        print("  执行笛卡尔路径...")
        arm.execute()
        print("  完成! 正方形已画完。")
    else:
        print(f"  规划覆盖率仅 {fraction*100:.1f}% — 路径不可达")

    # === 2. 圆弧运动 ===
    print("\n--- 2. 笛卡尔圆弧运动 ---")
    print("  末端画一个圆 (半径 5cm)")

    radius = 0.05
    num_points = 36  # 每 10° 一个路径点
    circle_waypoints = []

    for i in range(num_points + 1):
        angle = 2 * math.pi * i / num_points
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        circle_waypoints.append(make_pose(x, y, cz))

    arm.set_start_state_to_current_state()

    fraction = arm.compute_cartesian_path(
        waypoints=circle_waypoints,
        max_step=0.002,  # 2mm 插值
        jump_threshold=0.0,
    )

    if fraction > 0.9:
        print(f"  规划覆盖率: {fraction*100:.1f}%")
        print("  执行...")
        arm.execute()
        print("  完成! 圆已画完。")
    else:
        print(f"  规划覆盖率仅 {fraction*100:.1f}%")

    # === 3. 带姿态变化的直线 ===
    print("\n--- 3. 直线运动 + 末端旋转 ---")
    print("  沿 X 轴前进 15cm，同时末端绕 Z 轴旋转 90°")

    rotate_waypoints = []
    dist = 0.15
    num_pts = 20

    for i in range(num_pts + 1):
        t = i / num_pts
        x = cx + dist * t
        # 绕 Z 轴旋转: quaternion
        angle = math.pi / 2 * t  # 0 → 90°
        qw = math.cos(angle / 2)
        qz = math.sin(angle / 2)
        # 组合: 基础朝下 (qy=0.707, qw=0.707) + Z旋转
        rotate_waypoints.append(make_pose(x, cy, cz, qx=0.0, qy=0.707*qw,
                                          qz=0.707*qz, qw=0.707*qw))

    arm.set_start_state_to_current_state()
    fraction = arm.compute_cartesian_path(
        waypoints=rotate_waypoints,
        max_step=0.005,
        jump_threshold=0.0,
    )

    if fraction > 0.9:
        print(f"  规划覆盖率: {fraction*100:.1f}%")
        arm.execute()
        print("  完成!")
    else:
        print(f"  规划覆盖率不足: {fraction*100:.1f}%")

    print("\n所有笛卡尔运动完成!")
    moveit.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
