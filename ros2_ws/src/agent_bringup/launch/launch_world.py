from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
import os

from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Path to the URDF and the Gazebo launch files:
    pkg_path = get_package_share_directory("agent_bringup")
    gz_sim_path = get_package_share_directory("ros_gz_sim")
    gazebo_launch_file_dir = PathJoinSubstitution([gz_sim_path, "launch", "gz_sim.launch.py"])
    gui_path = os.path.join(pkg_path, "config", "gui.config")

    # define the launch arguments:
    world = PathJoinSubstitution([pkg_path, "worlds", LaunchConfiguration("world")])
    world_arg = DeclareLaunchArgument(
        name = "world",
        default_value = "empty_world.sdf",
        description = "Name of the world to be loaded, defaulting to empty_world.sdf"
    )

    # define the nodes to be launched:
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([gazebo_launch_file_dir]),
        launch_arguments={"gz_args": [f"--gui-config {gui_path} -r -v1 ", world]}.items()
        # launch_arguments={"gz_args": ["-r -v1 ", world]}.items()
    )

    return LaunchDescription([
        world_arg,
        gazebo
    ])