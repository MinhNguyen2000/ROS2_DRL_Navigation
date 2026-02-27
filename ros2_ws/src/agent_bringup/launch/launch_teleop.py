import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory

# what does this file launch?
# - teleop_twist_keyboard

def generate_launch_description():
    # define the launch arguments:
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value = "true",
        description = "Whether or not to use sim time, defaulting to false."
    )

    stamped = LaunchConfiguration("stamped")
    stamped_arg = DeclareLaunchArgument(
        "stamped",
        default_value = "true",
        description = "Whether or not to stamp the cmd_vel topic"
    )

    # define the nodes to be launched:
    teleop = Node(
        package = "teleop_twist_keyboard",
        executable = "teleop_twist_keyboard",
        name = "teleop_twist_keyboard",
        output = "screen",
        prefix = "xterm -e",
        parameters = [{"use_sim_time" : use_sim_time, "stamped" : stamped}]
    )

    return LaunchDescription([
        use_sim_time_arg,
        stamped_arg,
        teleop
    ])