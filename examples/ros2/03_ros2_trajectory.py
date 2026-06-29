"""
ROS2 轨迹执行 — 发送 FollowJointTrajectory action。

通过标准 control_msgs/FollowJointTrajectory action 接口发送轨迹，
兼容 MoveIt 等运动规划框架的输出。

前提:
    1. 启动机械臂节点:
       ros2 launch onero_ros2 bringup.launch.py can_channel:=can0
    2. 使能:
       ros2 service call /onero_arm_driver/enable std_srvs/srv/Trigger

运行:
    python examples/ros2/03_ros2_trajectory.py
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory


class TrajectoryClient(Node):
    def __init__(self):
        super().__init__("trajectory_client")
        self._action_client = ActionClient(
            self, FollowJointTrajectory,
            "/arm_controller/follow_joint_trajectory")

        self.get_logger().info("等待 FollowJointTrajectory action server...")
        self._action_client.wait_for_server()
        self.get_logger().info("已连接!")

    def send_trajectory(self, points: list[dict]) -> bool:
        """
        发送轨迹。

        Args:
            points: 列表, 每个元素 {"positions": [...], "time": seconds}
        """
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = [
            f"joint{i+1}-a1_r" for i in range(7)
        ]

        for pt in points:
            point = JointTrajectoryPoint()
            point.positions = pt["positions"]
            point.velocities = [0.0] * 7
            secs = int(pt["time"])
            nsecs = int((pt["time"] - secs) * 1e9)
            point.time_from_start = Duration(sec=secs, nanosec=nsecs)
            goal.trajectory.points.append(point)

        self.get_logger().info(f"发送轨迹: {len(points)} 个路径点")

        future = self._action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("轨迹被拒绝!")
            return False

        self.get_logger().info("轨迹已接受, 执行中...")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result().result
        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().info("轨迹执行完成!")
            return True
        else:
            self.get_logger().error(f"轨迹执行失败: error_code={result.error_code}")
            return False


def main():
    rclpy.init()
    node = TrajectoryClient()

    # 示例: 画一个简单的"挥手"轨迹
    # 固定前4个关节，末端2个关节来回摆动
    base_pos = [0.0, -0.5, 0.0, -1.0, 0.0, 0.5, 0.0]

    trajectory_points = []
    num_points = 10
    duration = 5.0  # 总时长 5 秒

    for i in range(num_points + 1):
        t = i / num_points
        pos = base_pos.copy()
        # joint_6 和 joint_7 做正弦摆动
        pos[5] = 0.5 + 0.5 * math.sin(2 * math.pi * t)
        pos[6] = 0.3 * math.sin(2 * math.pi * t)
        trajectory_points.append({
            "positions": pos,
            "time": t * duration,
        })

    # 如果第一个点时间为 0, 设为一个小值
    if trajectory_points[0]["time"] == 0.0:
        trajectory_points[0]["time"] = 0.01

    print("ROS2 轨迹执行示例")
    print(f"发送 {len(trajectory_points)} 个路径点, 总时长 {duration}s")
    print("关节 6-7 做正弦摆动 ('挥手')\n")

    try:
        success = node.send_trajectory(trajectory_points)
        if success:
            print("\n轨迹执行成功!")
        else:
            print("\n轨迹执行失败")
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
