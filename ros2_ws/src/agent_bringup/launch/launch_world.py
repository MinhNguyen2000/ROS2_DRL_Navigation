import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Path to the URDF and the Gazebo launch files
    gazebo_launch_file_dir = os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')

    default_world = os.path.join(
        get_package_share_directory('agent_bringup'),
        'worlds',
        'world_1.sdf'
    )

    world_arg = DeclareLaunchArgument(
        name='world',
        default_value=default_world,
        description="Path of the world to load"
    )

    world = LaunchConfiguration('world')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([gazebo_launch_file_dir]),
        launch_arguments={'gz_args': ['-r -v4 ', world]}.items()
    )

    return LaunchDescription([
        world_arg,
        gazebo
    ])