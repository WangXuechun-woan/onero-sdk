"""
ROS2 基础 — 订阅 /joint_states 实时显示关节状态。

前提:
    1. 启动机械臂节点:
       ros2 launch onero_ros2 bringup.launch.py can_channel:=can0
    2. 使能:
       ros2 service call /onero_arm_driver/enable std_srvs/srv/Trigger

运行:
    python examples/ros2/01_ros2_joint_states.py
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class JointStateViewer(Node):
    def __init__(self):
        super().__init__("joint_state_viewer")
        self.subscription = self.create_subscription(
            JointState, "/joint_states", self.callback, 10)
        self.count = 0

    def callback(self, msg: JointState):
        self.count += 1
        if self.count % 50 == 0:  # 每 50 帧打印一次 (约 0.25s)
            pos = [f"{p:.3f}" for p in msg.position]
            vel = [f"{v:.3f}" for v in msg.velocity]
            self.get_logger().info(f"位置: {pos}")
            self.get_logger().info(f"速度: {vel}")
            print()


def main():
    rclpy.init()
    node = JointStateViewer()
    print("订阅 /joint_states...")
    print("(确保 onero_ros2 节点已启动并使能)\n")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
