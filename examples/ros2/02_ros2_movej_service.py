"""
ROS2 服务调用 — 通过 service 发送 movej 指令。

演示如何通过 ROS2 service 接口控制机械臂运动到目标位置。

前提:
    1. 启动机械臂节点:
       ros2 launch onero_ros2 bringup.launch.py can_channel:=can0
    2. 使能:
       ros2 service call /onero_arm_driver/enable std_srvs/srv/Trigger

运行:
    python examples/ros2/02_ros2_movej_service.py
"""

import math
import sys

import rclpy
from rclpy.node import Node
from onero_msgs.srv import MoveJ


class MovejClient(Node):
    def __init__(self):
        super().__init__("movej_client")
        self.client = self.create_client(MoveJ, "/onero_arm_driver/movej")
        while not self.client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("等待 movej 服务...")

    def send_movej(self, joints: list[float], speed: float = 1.0,
                   acceleration: float = 2.0) -> bool:
        """发送 movej 请求并等待结果。"""
        req = MoveJ.Request()
        req.positions = joints
        req.speed = speed
        req.acceleration = acceleration

        self.get_logger().info(
            f"发送 movej: {[f'{j:.2f}' for j in joints]}, "
            f"speed={speed}, acc={acceleration}")

        future = self.client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        result = future.result()
        if result.success:
            self.get_logger().info("movej 完成!")
        else:
            self.get_logger().error(f"movej 失败: {result.message}")
        return result.success


def main():
    rclpy.init()
    node = MovejClient()

    # 定义几个目标位姿 (弧度)
    waypoints = [
        # 零位
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # 位置 A
        [0.5, -0.3, 0.2, -1.2, 0.0, 0.8, 0.0],
        # 位置 B
        [-0.5, 0.3, -0.2, -0.8, 0.5, 1.0, 0.3],
        # 回到零位
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]

    print("通过 ROS2 MoveJ Service 控制机械臂")
    print(f"将依次运动到 {len(waypoints)} 个目标位置\n")

    try:
        for i, target in enumerate(waypoints):
            print(f"--- 目标 {i+1}/{len(waypoints)} ---")
            success = node.send_movej(target, speed=0.8)
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
