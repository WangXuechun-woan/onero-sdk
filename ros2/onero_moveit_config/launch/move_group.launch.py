"""Launch MoveIt2 move_group for Onero A1 arm."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_fake_hardware_arg = DeclareLaunchArgument(
        "use_fake_hardware", default_value="false",
        description="Use mock hardware for testing without robot",
    )

    # Robot description from onero_ros2 package
    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name="xacro")]),
        " ",
        PathJoinSubstitution([
            FindPackageShare("onero_ros2"), "description",
            "a1_ros2_control.urdf.xacro",
        ]),
        " use_fake_hardware:=", LaunchConfiguration("use_fake_hardware"),
    ])
    robot_description = {"robot_description": robot_description_content}

    # SRDF
    robot_description_semantic_content = Command([
        "cat ",
        PathJoinSubstitution([
            FindPackageShare("onero_moveit_config"), "config", "a1.srdf",
        ]),
    ])
    robot_description_semantic = {
        "robot_description_semantic": robot_description_semantic_content,
    }

    # MoveIt config files
    moveit_config_pkg = FindPackageShare("onero_moveit_config")

    kinematics_yaml = PathJoinSubstitution([
        moveit_config_pkg, "config", "kinematics.yaml",
    ])
    ompl_planning_yaml = PathJoinSubstitution([
        moveit_config_pkg, "config", "ompl_planning.yaml",
    ])
    moveit_controllers_yaml = PathJoinSubstitution([
        moveit_config_pkg, "config", "moveit_controllers.yaml",
    ])
    joint_limits_yaml = PathJoinSubstitution([
        moveit_config_pkg, "config", "joint_limits.yaml",
    ])

    # MoveIt parameters
    moveit_config = [
        robot_description,
        robot_description_semantic,
        kinematics_yaml,
        ompl_planning_yaml,
        moveit_controllers_yaml,
        joint_limits_yaml,
        {"use_sim_time": False},
    ]

    # move_group node
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=moveit_config,
    )

    # RViz with MoveIt plugin
    rviz_config = PathJoinSubstitution([
        moveit_config_pkg, "config", "moveit.rviz",
    ])
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        output="log",
        arguments=["-d", rviz_config],
        parameters=[
            robot_description,
            robot_description_semantic,
            kinematics_yaml,
        ],
    )

    return LaunchDescription([
        use_fake_hardware_arg,
        move_group_node,
        rviz_node,
    ])
