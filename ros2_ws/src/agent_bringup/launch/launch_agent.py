from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory

# what does this file launch?
# - robot_state_publisher
# - gazebo
# - spawner

def generate_launch_description():
    # set the required paths:
    pkg_path = get_package_share_directory("agent_bringup")
    xacro_path = PathJoinSubstitution([pkg_path, "urdf", "agent.urdf.xacro"])
    gazebo_launch_path = PathJoinSubstitution([pkg_path, "launch", "launch_world.py"])

    # define the launch arguments:
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value = "true",
        description = "Whether or not to use sim time, defaulting to false."
    )

    world = PathJoinSubstitution([pkg_path, "worlds", LaunchConfiguration("world")])
    world_arg = DeclareLaunchArgument(
        "world",
        default_value = "empty_world.sdf",
        description = "Name of the world to be loaded, defaulting to empty_world.sdf"
    )

    # set the required parameters:
    agent_name = "agent"
    robot_description = Command(["xacro ", xacro_path, " agent_name:=", agent_name])
    rsp_parameters = {"robot_description": ParameterValue(robot_description, value_type = str), "use_sim_time" : use_sim_time}

    # define the nodes to be launched:
    rsp = Node(
        package = "robot_state_publisher",
        executable = "robot_state_publisher",
        name = "robot_state_publisher",
        namespace = agent_name,
        parameters = [rsp_parameters]
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([gazebo_launch_path]),
        launch_arguments = {"world" : world}.items()
    )

    spawner = Node(
        package = "ros_gz_sim",
        executable = "create",
        arguments = ["-topic", f"{agent_name}/robot_description",
                    "-name", agent_name,
                    "-z", "0.0"],
        output = "screen"
    )

    return LaunchDescription([
        use_sim_time_arg,
        world_arg,
        rsp, 
        gazebo,
        spawner
    ])
    
