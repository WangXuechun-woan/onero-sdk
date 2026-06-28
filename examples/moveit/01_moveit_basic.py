"""
MoveIt2 基础 — 规划并执行关节空间运动。

使用 MoveIt2 Python API (moveit_py) 进行运动规划和执行。
MoveIt 会自动处理碰撞检测、运动学约束和轨迹优化。

前提:
    1. 启动 MoveIt:
       ros2 launch onero_moveit_config move_group.launch.py
    2. 确保机械臂节点也已启动 (bringup 通常已包含)

运行:
    python examples/moveit/01_moveit_basic.py
"""

import math

import rclpy
from rclpy.node import Node
from moveit.planning import MoveItPy
from moveit.core.robot_state import RobotState


def main():
    rclpy.init()

    # 初始化 MoveIt
    print("初始化 MoveIt...")
    moveit = MoveItPy(node_name="moveit_basic_example")
    arm = moveit.get_planning_component("arm")

    robot_model = moveit.get_robot_model()
    robot_state = RobotState(robot_model)

    print("MoveIt 就绪!\n")

    # === 1. 运动到命名姿态 ===
    print("--- 1. 运动到 'home' 姿态 ---")
    arm.set_start_state_to_current_state()
    arm.set_goal_state(configuration_name="home")

    plan_result = arm.plan()
    if plan_result:
        print("  规划成功, 执行中...")
        arm.execute()
        print("  到达 home!")
    else:
        print("  规划失败!")

    # === 2. 运动到指定关节角度 ===
    print("\n--- 2. 运动到指定关节角度 ---")
    target_joints = {
        "joint_1": 0.5,
        "joint_2": -0.3,
        "joint_3": 0.2,
        "joint_4": -1.2,
        "joint_5": 0.0,
        "joint_6": 0.8,
        "joint_7": 0.0,
    }
    print(f"  目标: {target_joints}")

    arm.set_start_state_to_current_state()
    robot_state.set_joint_group_positions("arm", list(target_joints.values()))
    arm.set_goal_state(robot_state=robot_state)

    plan_result = arm.plan()
    if plan_result:
        print("  规划成功, 执行中...")
        arm.execute()
        print("  到达目标!")
    else:
        print("  规划失败 — 可能存在碰撞或超出关节限制")

    # === 3. 多目标连续运动 ===
    print("\n--- 3. 连续运动到多个目标 ---")
    waypoints = [
        [0.3, -0.5, 0.1, -0.9, 0.2, 0.6, 0.1],
        [-0.3, -0.2, -0.1, -1.5, -0.2, 1.2, -0.1],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 零位
    ]

    for i, joints in enumerate(waypoints):
        print(f"  目标 {i+1}/{len(waypoints)}...")
        arm.set_start_state_to_current_state()
        robot_state.set_joint_group_positions("arm", joints)
        arm.set_goal_state(robot_state=robot_state)

        plan_result = arm.plan()
        if plan_result:
            arm.execute()
            print(f"    到达!")
        else:
            print(f"    规划失败, 跳过")

    print("\n完成!")
    moveit.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
