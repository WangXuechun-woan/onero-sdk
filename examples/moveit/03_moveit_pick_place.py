"""
MoveIt2 抓取放置 — Pick & Place 任务。

演示一个完整的抓取-放置任务流程:
  1. 运动到预抓取位
  2. 直线下降到抓取点
  3. 闭合夹爪
  4. 直线抬起
  5. 运动到放置位
  6. 直线下降
  7. 松开夹爪
  8. 抬起并回到 home

前提:
    ros2 launch onero_moveit_config move_group.launch.py

运行:
    python examples/moveit/03_moveit_pick_place.py
"""

import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point, Quaternion
from moveit.planning import MoveItPy
from moveit.core.robot_state import RobotState


# --- 夹爪控制 (模拟) ---

class GripperController:
    """简单的夹爪控制封装。实际项目中替换为真实的夹爪驱动。"""

    def __init__(self, node: Node):
        self._node = node
        self._is_open = True

    def open(self):
        self._node.get_logger().info("夹爪: 打开")
        self._is_open = True
        time.sleep(0.5)  # 等待夹爪动作完成

    def close(self):
        self._node.get_logger().info("夹爪: 关闭 (夹紧)")
        self._is_open = False
        time.sleep(0.5)


# --- 辅助函数 ---

def make_pose(x, y, z, qx=0.0, qy=1.0, qz=0.0, qw=0.0) -> Pose:
    """创建末端朝下的位姿。"""
    pose = Pose()
    pose.position = Point(x=x, y=y, z=z)
    pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
    return pose


def cartesian_move(arm, waypoints, description=""):
    """执行笛卡尔路径。"""
    if description:
        print(f"  {description}")

    arm.set_start_state_to_current_state()
    fraction = arm.compute_cartesian_path(
        waypoints=waypoints,
        max_step=0.005,
        jump_threshold=0.0,
    )

    if fraction > 0.9:
        arm.execute()
        return True
    else:
        print(f"    路径规划失败 (覆盖率: {fraction*100:.1f}%)")
        return False


# --- 主程序 ---

def main():
    rclpy.init()

    print("初始化 MoveIt...")
    moveit = MoveItPy(node_name="moveit_pick_place_example")
    arm = moveit.get_planning_component("arm")
    node = moveit.get_node()

    gripper = GripperController(node)
    gripper.open()

    print("MoveIt 就绪!\n")

    # 定义抓取和放置位置
    pick_x, pick_y = 0.35, -0.10   # 抓取点 XY
    place_x, place_y = 0.35, 0.10  # 放置点 XY
    table_z = 0.05                   # 桌面高度
    approach_z = 0.20                # 预抓取高度
    grasp_z = table_z + 0.02         # 夹取高度 (桌面上方 2cm)
    lift_z = 0.20                    # 抬起高度

    print(f"抓取位: ({pick_x}, {pick_y})")
    print(f"放置位: ({place_x}, {place_y})")
    print(f"桌面高度: {table_z}m\n")

    # === Step 1: 回到 home ===
    print("--- Step 1: 回到 home ---")
    arm.set_start_state_to_current_state()
    arm.set_goal_state(configuration_name="home")
    result = arm.plan()
    if result:
        arm.execute()
        print("  到达 home\n")
    else:
        print("  规划失败!")
        return

    # === Step 2: 运动到预抓取位 ===
    print("--- Step 2: 到预抓取位 ---")
    pre_pick = make_pose(pick_x, pick_y, approach_z)
    arm.set_start_state_to_current_state()
    arm.set_goal_state(pose_stamped_msg=pre_pick, pose_link="tool0")
    result = arm.plan()
    if result:
        arm.execute()
        print("  到达预抓取位\n")
    else:
        print("  规划失败!")
        return

    # === Step 3: 直线下降到夹取位 ===
    print("--- Step 3: 直线下降 (抓取) ---")
    grasp_pose = make_pose(pick_x, pick_y, grasp_z)
    if not cartesian_move(arm, [grasp_pose], "下降到夹取点"):
        return
    print()

    # === Step 4: 闭合夹爪 ===
    print("--- Step 4: 闭合夹爪 ---")
    gripper.close()
    print()

    # === Step 5: 直线抬起 ===
    print("--- Step 5: 抬起物体 ---")
    lift_pose = make_pose(pick_x, pick_y, lift_z)
    if not cartesian_move(arm, [lift_pose], "直线抬起"):
        return
    print()

    # === Step 6: 运动到预放置位 ===
    print("--- Step 6: 到预放置位 ---")
    pre_place = make_pose(place_x, place_y, lift_z)
    arm.set_start_state_to_current_state()
    arm.set_goal_state(pose_stamped_msg=pre_place, pose_link="tool0")
    result = arm.plan()
    if result:
        arm.execute()
        print("  到达预放置位\n")
    else:
        print("  规划失败!")
        return

    # === Step 7: 直线下降到放置位 ===
    print("--- Step 7: 直线下降 (放置) ---")
    place_pose = make_pose(place_x, place_y, grasp_z)
    if not cartesian_move(arm, [place_pose], "下降到放置点"):
        return
    print()

    # === Step 8: 打开夹爪 ===
    print("--- Step 8: 松开夹爪 ---")
    gripper.open()
    print()

    # === Step 9: 抬起并回 home ===
    print("--- Step 9: 抬起并回 home ---")
    retreat_pose = make_pose(place_x, place_y, approach_z)
    cartesian_move(arm, [retreat_pose], "抬起")

    arm.set_start_state_to_current_state()
    arm.set_goal_state(configuration_name="home")
    result = arm.plan()
    if result:
        arm.execute()

    print("\n抓取放置任务完成!")
    gripper.open()
    moveit.shutdown()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
