"""
Onero A1 ROS2 Control Node — standalone arm driver.

Bridges the Onero SDK to ROS2 without requiring C++ pluginlib.
Runs a 200Hz control loop internally and exposes standard ROS2 interfaces:
  - Publishes: /joint_states (sensor_msgs/JointState)
  - Action server: /arm_controller/follow_joint_trajectory
  - Service: ~/enable, ~/disable

This approach avoids the ros2_control pluginlib limitation (C++ only)
while still being compatible with MoveIt2 via FollowJointTrajectory action.
"""

import asyncio
import threading
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from std_srvs.srv import Trigger

from onero_sdk.backend import ArmState
from onero_sdk.can_backend import CanBackend
from onero_sdk.model_registry import get_model


class OneroArmNode(Node):
    """Standalone ROS2 node driving the Onero arm at 200Hz."""

    JOINT_NAMES = [
        "joint1-a1_r", "joint2-a1_r", "joint3-a1_r", "joint4-a1_r",
        "joint5-a1_r", "joint6-a1_r", "joint7-a1_r",
    ]

    def __init__(self):
        super().__init__("onero_arm_driver")

        # Parameters
        self.declare_parameter("can_channel", "can0")
        self.declare_parameter("model", "A1R")
        self.declare_parameter("kp", 50.0)
        self.declare_parameter("kd", 1.0)
        self.declare_parameter("control_rate", 200.0)

        self._can_channel = self.get_parameter("can_channel").value
        self._model_name = self.get_parameter("model").value
        self._kp = self.get_parameter("kp").value
        self._kd = self.get_parameter("kd").value
        self._control_rate = self.get_parameter("control_rate").value
        self._control_dt = 1.0 / self._control_rate

        # Hardware state
        model_info = get_model(self._model_name)
        self._n_joints = len(model_info.joints)
        self._backend: CanBackend = None
        self._enabled = False
        self._lock = threading.Lock()

        # Optional gravity compensation
        self._dynamics = None
        try:
            from onero_sdk.dynamics import DynamicsModel
            from onero_sdk.model_registry import _find_urdf
            urdf_path = _find_urdf(self._model_name)
            if urdf_path:
                self._dynamics = DynamicsModel(str(urdf_path))
                self.get_logger().info("Gravity compensation enabled (pinocchio)")
        except ImportError:
            self.get_logger().info(
                "No gravity compensation (install onero-sdk[dynamics] to enable)")

        # State arrays
        self._hw_positions = np.zeros(self._n_joints)
        self._hw_velocities = np.zeros(self._n_joints)
        self._hw_efforts = np.zeros(self._n_joints)

        # Command target (position hold by default)
        self._cmd_positions = np.zeros(self._n_joints)
        self._cmd_velocities = np.zeros(self._n_joints)

        # Trajectory execution state
        self._executing_trajectory = False
        self._trajectory_points = []
        self._trajectory_start_time = 0.0
        self._trajectory_joint_indices = []

        # Publishers
        self._joint_state_pub = self.create_publisher(
            JointState, "/joint_states", 10
        )

        # Action server for MoveIt2 compatibility
        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            "/arm_controller/follow_joint_trajectory",
            execute_callback=self._execute_trajectory,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=ReentrantCallbackGroup(),
        )

        # Services
        self._enable_srv = self.create_service(
            Trigger, "~/enable", self._enable_callback
        )
        self._disable_srv = self.create_service(
            Trigger, "~/disable", self._disable_callback
        )

        # Connect to hardware
        try:
            self._backend = CanBackend(
                channel=self._can_channel,
                joint_specs=model_info.joints,
                recv_timeout=0.003,
            )
            self.get_logger().info(
                f"Connected to CAN '{self._can_channel}', model={self._model_name}"
            )
        except Exception as e:
            self.get_logger().fatal(f"Failed to connect: {e}")
            raise

        # Control loop thread
        self._running = True
        self._control_thread = threading.Thread(
            target=self._control_loop, daemon=True
        )
        self._control_thread.start()

        self.get_logger().info("Onero arm driver started (call ~/enable to activate)")

    def destroy_node(self):
        self._running = False
        if self._control_thread.is_alive():
            self._control_thread.join(timeout=1.0)
        if self._backend:
            if self._enabled:
                self._backend.disable()
            self._backend.close()
        super().destroy_node()

    # ─── Control Loop (200Hz) ──────────────────────────────────────────

    def _control_loop(self):
        """Main 200Hz loop: compute command, send, update state from response."""
        while self._running:
            t0 = time.monotonic()

            with self._lock:
                if not self._enabled:
                    time.sleep(self._control_dt)
                    continue

                # Compute command
                if self._executing_trajectory:
                    self._update_trajectory_command()

                # Send command and get state from response
                try:
                    torques_ff = [0.0] * self._n_joints
                    if self._dynamics:
                        try:
                            torques_ff = self._dynamics.gravity_torques(
                                self._cmd_positions.tolist())
                        except Exception:
                            pass
                    state = self._backend.command(
                        positions=self._cmd_positions.tolist(),
                        velocities=self._cmd_velocities.tolist(),
                        kps=[self._kp] * self._n_joints,
                        kds=[self._kd] * self._n_joints,
                        torques=torques_ff,
                    )
                    self._hw_positions[:] = state.joints
                    self._hw_velocities[:] = state.velocities
                    self._hw_efforts[:] = state.torques
                except Exception as e:
                    self.get_logger().error(f"Control cycle failed: {e}")
                    time.sleep(self._control_dt)
                    continue

            # Publish joint states (outside lock)
            self._publish_joint_state()

            # Rate control
            elapsed = time.monotonic() - t0
            sleep_time = self._control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _publish_joint_state(self):
        """Publish current joint state."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.JOINT_NAMES
        msg.position = self._hw_positions.tolist()
        msg.velocity = self._hw_velocities.tolist()
        msg.effort = self._hw_efforts.tolist()
        self._joint_state_pub.publish(msg)

    # ─── Trajectory Execution ──────────────────────────────────────────

    def _update_trajectory_command(self):
        """Interpolate trajectory to current time."""
        elapsed = time.monotonic() - self._trajectory_start_time
        points = self._trajectory_points

        if not points:
            self._executing_trajectory = False
            return

        # Find bracketing points
        last_point = points[-1]
        last_time = last_point.time_from_start.sec + last_point.time_from_start.nanosec * 1e-9

        if elapsed >= last_time:
            # Trajectory complete — hold final position
            for i, ji in enumerate(self._trajectory_joint_indices):
                self._cmd_positions[ji] = last_point.positions[i]
                self._cmd_velocities[ji] = 0.0
            self._executing_trajectory = False
            return

        # Linear interpolation between points
        prev_point = None
        prev_time = 0.0
        for pt in points:
            pt_time = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
            if pt_time >= elapsed:
                # Interpolate between prev_point and pt
                if prev_point is None:
                    alpha = 0.0
                else:
                    dt_seg = pt_time - prev_time
                    alpha = (elapsed - prev_time) / dt_seg if dt_seg > 0 else 1.0

                for i, ji in enumerate(self._trajectory_joint_indices):
                    p0 = prev_point.positions[i] if prev_point else self._hw_positions[ji]
                    p1 = pt.positions[i]
                    self._cmd_positions[ji] = p0 + alpha * (p1 - p0)

                    if pt.velocities:
                        v0 = prev_point.velocities[i] if prev_point and prev_point.velocities else 0.0
                        v1 = pt.velocities[i]
                        self._cmd_velocities[ji] = v0 + alpha * (v1 - v0)
                    else:
                        self._cmd_velocities[ji] = 0.0
                return

            prev_point = pt
            prev_time = pt_time

    def _goal_callback(self, goal_request):
        """Accept trajectory goals."""
        if not self._enabled:
            self.get_logger().warn("Rejecting trajectory: arm not enabled")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle):
        """Allow cancellation — hold current position."""
        self.get_logger().info("Trajectory cancelled, holding position")
        return CancelResponse.ACCEPT

    async def _execute_trajectory(self, goal_handle):
        """Execute FollowJointTrajectory action."""
        trajectory = goal_handle.request.trajectory
        joint_names = trajectory.joint_names
        points = trajectory.points

        if not points:
            goal_handle.abort()
            result = FollowJointTrajectory.Result()
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            return result

        # Map trajectory joint names to our indices
        joint_indices = []
        for name in joint_names:
            if name in self.JOINT_NAMES:
                joint_indices.append(self.JOINT_NAMES.index(name))
            else:
                self.get_logger().error(f"Unknown joint: {name}")
                goal_handle.abort()
                result = FollowJointTrajectory.Result()
                result.error_code = FollowJointTrajectory.Result.INVALID_JOINTS
                return result

        # Start trajectory execution
        with self._lock:
            self._trajectory_points = points
            self._trajectory_joint_indices = joint_indices
            self._trajectory_start_time = time.monotonic()
            self._executing_trajectory = True

        # Wait for completion
        last_time = (
            points[-1].time_from_start.sec
            + points[-1].time_from_start.nanosec * 1e-9
        )

        while self._executing_trajectory:
            if goal_handle.is_cancel_requested:
                with self._lock:
                    self._executing_trajectory = False
                    self._cmd_positions[:] = self._hw_positions
                    self._cmd_velocities[:] = 0.0
                goal_handle.canceled()
                return FollowJointTrajectory.Result()

            # Publish feedback
            feedback = FollowJointTrajectory.Feedback()
            feedback.joint_names = joint_names
            feedback.actual.positions = self._hw_positions[joint_indices].tolist()
            feedback.actual.velocities = self._hw_velocities[joint_indices].tolist()
            feedback.desired.positions = self._cmd_positions[joint_indices].tolist()
            goal_handle.publish_feedback(feedback)

            await asyncio.sleep(0.05)

        # Check final error
        final_error = np.abs(
            self._hw_positions[joint_indices]
            - np.array(points[-1].positions)
        )
        max_error = float(final_error.max())

        result = FollowJointTrajectory.Result()
        if max_error < 0.1:  # 0.1 rad tolerance
            goal_handle.succeed()
            result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        else:
            self.get_logger().warn(f"Trajectory goal tolerance exceeded: {max_error:.3f} rad")
            goal_handle.succeed()
            result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        return result

    # ─── Services ──────────────────────────────────────────────────────

    def _enable_callback(self, request, response):
        """Enable motors."""
        try:
            with self._lock:
                state = self._backend.enable()
                self._hw_positions[:] = state.joints
                self._cmd_positions[:] = state.joints
                self._cmd_velocities[:] = 0.0
                self._enabled = True
            response.success = True
            response.message = "Motors enabled"
            self.get_logger().info("Motors enabled")
        except Exception as e:
            response.success = False
            response.message = str(e)
            self.get_logger().error(f"Enable failed: {e}")
        return response

    def _disable_callback(self, request, response):
        """Disable motors."""
        try:
            with self._lock:
                self._executing_trajectory = False
                self._backend.disable()
                self._enabled = False
            response.success = True
            response.message = "Motors disabled"
            self.get_logger().info("Motors disabled")
        except Exception as e:
            response.success = False
            response.message = str(e)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = OneroArmNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
