"""Launch Onero A1 arm with real hardware (CanBackend)."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessExit
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
        description="Default position gain (MIT impedance)",
    )
    kd_arg = DeclareLaunchArgument(
        "kd", default_value="1.0",
        description="Default damping gain (MIT impedance)",
    )

    # Generate URDF from xacro
    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name="xacro")]),
        " ",
        PathJoinSubstitution([
            FindPackageShare("onero_ros2"), "description",
            "a1_ros2_control.urdf.xacro",
        ]),
        " can_channel:=", LaunchConfiguration("can_channel"),
        " kp:=", LaunchConfiguration("kp"),
        " kd:=", LaunchConfiguration("kd"),
        " use_fake_hardware:=false",
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

    # Spawn controllers after control_node starts
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

    # Chain: wait for joint_state_broadcaster, then spawn arm_controller
    delay_arm_controller = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[arm_controller_spawner],
        )
    )

    return LaunchDescription([
        can_channel_arg,
        kp_arg,
        kd_arg,
        control_node,
        robot_state_pub,
        joint_state_broadcaster_spawner,
        delay_arm_controller,
    ])
