import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction, OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.actions import Node, PushRosNamespace
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory

# what does this file launch?
# - launch_agent launch file
# - drl_gui node

def generate_launch_description():
    pkg_path = get_package_share_directory("agent_bringup")
    agent_launch_path = PathJoinSubstitution([pkg_path, "launch", "launch_agent.py"])
    rviz_launch_path = PathJoinSubstitution([pkg_path, "launch", "launch_rviz.py"])

    # define the launch arguments
    world = LaunchConfiguration("world")
    world_arg = DeclareLaunchArgument(
        "world",
        default_value = "world_1.sdf",
        description = "Name of the world to be loaded"
    )

    use_sim_time = LaunchConfiguration("use_sim_time")
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value = "true",
        description = "Whether or not to use sim time, defaulting to false."
    )

    use_ros_control = LaunchConfiguration("use_ros_control")
    use_ros_control_arg = DeclareLaunchArgument(
        "use_ros_control",
        default_value = "true",
        description = "If true, control the agent using ros2_control, otherwise, use Gazebo plugins. Defaults to false"
    )

    def launch_setup(context):
        world_str = LaunchConfiguration("world").perform(context)
        world_name = world_str.split('.')[0]

        launch_rviz = IncludeLaunchDescription(
            PythonLaunchDescriptionSource([rviz_launch_path])
        )

        x_offset = -3.0 if world_name == "world_1" else -1.0
        y_offset = x_offset

        launch_agent = IncludeLaunchDescription(
            PythonLaunchDescriptionSource([agent_launch_path]),
            launch_arguments = {
                "world": world,
                "use_sim_time": use_sim_time, 
                "use_ros_control": use_ros_control,
                "x_offset": str(x_offset),
                "y_offset": str(y_offset)
            }.items()
        )

        drl_gui = Node(
            package = 'drl_gui',
            executable = 'gui_node',
            name = 'drl_gui_node',
            arguments = [{
                'x_offset': x_offset,
                'y_offset': y_offset
            }]
        )

        return [
            launch_agent,
            launch_rviz,
            drl_gui
        ]

    return LaunchDescription([
        world_arg,
        use_sim_time_arg,
        use_ros_control_arg,
        OpaqueFunction(function=launch_setup)
    ])
    
