"""
ROS2 单点运动 — 通过 FollowJointTrajectory action 发送 movej 指令。

演示如何通过 ROS2 action 接口控制机械臂运动到目标位置。
发送单个路径点的轨迹等价于 movej。

前提:
    1. 启动机械臂节点:
       ros2 launch onero_ros2 bringup.launch.py can_channel:=can0
    2. 使能:
       ros2 service call /onero_arm_driver/enable std_srvs/srv/Trigger

运行:
    python examples/ros2/02_ros2_movej_service.py
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory

JOINT_NAMES = [
    "joint1-a1_r", "joint2-a1_r", "joint3-a1_r", "joint4-a1_r",
    "joint5-a1_r", "joint6-a1_r", "joint7-a1_r",
]


class MovejClient(Node):
    def __init__(self):
        super().__init__("movej_client")
        self._action_client = ActionClient(
            self, FollowJointTrajectory,
            "/arm_controller/follow_joint_trajectory")

        self.get_logger().info("等待 FollowJointTrajectory action server...")
        self._action_client.wait_for_server()
        self.get_logger().info("已连接!")

    def send_movej(self, joints: list[float], duration: float = 2.0) -> bool:
        """发送单点轨迹 (等价于 movej)。"""
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = JOINT_NAMES

        point = JointTrajectoryPoint()
        point.positions = joints
        point.velocities = [0.0] * 7
        secs = int(duration)
        nsecs = int((duration - secs) * 1e9)
        point.time_from_start = Duration(sec=secs, nanosec=nsecs)
        goal.trajectory.points = [point]

        self.get_logger().info(
            f"发送 movej: {[f'{j:.2f}' for j in joints]}, "
            f"duration={duration}s")

        future = self._action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("轨迹被拒绝!")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result().result
        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().info("movej 完成!")
            return True
        else:
            self.get_logger().error(f"movej 失败: error_code={result.error_code}")
            return False


def main():
    rclpy.init()
    node = MovejClient()

    waypoints = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, -0.3, 0.2, -1.2, 0.0, 0.8, 0.0],
        [-0.5, 0.3, -0.2, -0.8, 0.5, 1.0, 0.3],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]

    print("通过 ROS2 FollowJointTrajectory Action 控制机械臂")
    print(f"将依次运动到 {len(waypoints)} 个目标位置\n")

    try:
        for i, target in enumerate(waypoints):
            print(f"--- 目标 {i+1}/{len(waypoints)} ---")
            success = node.send_movej(target, duration=2.0)
            if not success:
                print("运动失败，中止")
                break
            print()

        print("所有目标完成!")

    except KeyboardInterrupt:
        print("\n用户中断")

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
