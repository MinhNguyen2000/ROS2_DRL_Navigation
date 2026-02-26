from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory

# what does this file launch?
# - rviz

def generate_launch_description():
    # define the launch arguments:
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value = "true",
        description = "Whether or not to use sim time, defaulting to false."
    )

    # set the required parameters:
    agent_name = "agent"

    # nodes:
    rviz = Node(
        package = "rviz2",
        executable = "rviz2",
        name = "rviz2",
        namespace = agent_name, 
        output = "screen",
        parameters = [{"use_sim_time" : use_sim_time}]
    )

    return LaunchDescription([
        use_sim_time_arg,
        rviz
    ])



