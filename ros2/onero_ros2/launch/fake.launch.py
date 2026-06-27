"""Launch Onero A1 arm with mock hardware (no real robot needed).

Useful for testing MoveIt2 integration without hardware.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import (
    Command, FindExecutable, PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Generate URDF from xacro with fake hardware
    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name="xacro")]),
        " ",
        PathJoinSubstitution([
            FindPackageShare("onero_ros2"), "description",
            "a1_ros2_control.urdf.xacro",
        ]),
        " use_fake_hardware:=true",
        " fake_sensor_commands:=true",
    ])
    robot_description = {"robot_description": robot_description_content}

    # Controller manager config
    controllers_config = PathJoinSubstitution([
        FindPackageShare("onero_ros2"), "config", "controllers.yaml",
    ])

    # Nodes
    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[robot_description, controllers_config],
        output="both",
    )

    robot_state_pub = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[robot_description],
        output="both",
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster"],
    )

    arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["arm_controller"],
    )

    delay_arm_controller = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[arm_controller_spawner],
        )
    )

    return LaunchDescription([
        control_node,
        robot_state_pub,
        joint_state_broadcaster_spawner,
        delay_arm_controller,
    ])
