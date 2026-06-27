"""Launch Onero A1 arm driver node with real hardware."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Arguments
    can_channel_arg = DeclareLaunchArgument(
        "can_channel", default_value="can0",
        description="SocketCAN interface name",
    )
    kp_arg = DeclareLaunchArgument(
        "kp", default_value="50.0",
        description="Position gain (MIT impedance)",
    )
    kd_arg = DeclareLaunchArgument(
        "kd", default_value="1.0",
        description="Damping gain (MIT impedance)",
    )

    # Generate URDF from xacro
    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name="xacro")]),
        " ",
        PathJoinSubstitution([
            FindPackageShare("onero_ros2"), "description",
            "a1_ros2_control.urdf.xacro",
        ]),
        " use_fake_hardware:=false",
    ])
    robot_description = {"robot_description": robot_description_content}

    # Arm driver node (standalone — no ros2_control needed)
    arm_node = Node(
        package="onero_ros2",
        executable="arm_node",
        name="onero_arm_driver",
        parameters=[{
            "can_channel": LaunchConfiguration("can_channel"),
            "kp": LaunchConfiguration("kp"),
            "kd": LaunchConfiguration("kd"),
        }],
        output="both",
    )

    # Robot state publisher (TF from joint_states)
    robot_state_pub = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[robot_description],
        output="both",
    )

    return LaunchDescription([
        can_channel_arg,
        kp_arg,
        kd_arg,
        arm_node,
        robot_state_pub,
    ])
