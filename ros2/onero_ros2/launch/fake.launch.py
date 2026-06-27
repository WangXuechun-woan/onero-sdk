"""Launch Onero A1 in mock mode (no hardware, for MoveIt2 testing).

Uses a simulated arm_node that publishes fake joint_states.
"""

from launch import LaunchDescription
from launch.substitutions import (
    Command, FindExecutable, PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Generate URDF from xacro (with fake hardware flag for reference)
    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name="xacro")]),
        " ",
        PathJoinSubstitution([
            FindPackageShare("onero_ros2"), "description",
            "a1_ros2_control.urdf.xacro",
        ]),
        " use_fake_hardware:=true",
    ])
    robot_description = {"robot_description": robot_description_content}

    # Joint state publisher GUI (for manually moving joints in RViz)
    joint_state_publisher = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
        output="screen",
    )

    # Robot state publisher (TF)
    robot_state_pub = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[robot_description],
        output="both",
    )

    return LaunchDescription([
        robot_state_pub,
        joint_state_publisher,
    ])
