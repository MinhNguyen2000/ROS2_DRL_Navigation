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
# - robot_state_publisher
# - gazebo
# - spawner
# - ros_gz_bridge
# - diff_drive_spawner
# - joint_broadcaster_spawner
# - laser_scan_matcher

def generate_launch_description():
    # set the required paths:
    pkg_path = get_package_share_directory("agent_bringup")
    xacro_path = PathJoinSubstitution([pkg_path, "urdf", "agent.urdf.xacro"])
    gazebo_launch_path = PathJoinSubstitution([pkg_path, "launch", "launch_world.py"])
    bridge_path_1 = os.path.join(pkg_path, "config", "gz_bridge_ros_control.yaml")
    bridge_path_2 = os.path.join(pkg_path, "config", "gz_bridge_gazebo_control.yaml")

    # define the launch arguments:
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value = "true",
        description = "Whether or not to use sim time, defaulting to false."
    )

    use_ros_control = LaunchConfiguration("use_ros_control")
    use_ros_control_arg = DeclareLaunchArgument(
        "use_ros_control",
        default_value = "false",
        description = "If true, control the agent using ros2_control, otherwise, use Gazebo plugins. Defaults to false"
    )

    world = PathJoinSubstitution([pkg_path, "worlds", LaunchConfiguration("world")])
    world_arg = DeclareLaunchArgument(
        "world",
        default_value = "empty_world.sdf",
        description = "Name of the world to be loaded, defaulting to empty_world.sdf"
    )

    # set the required parameters:
    agent_name = "agent"
    
    robot_description = Command(["xacro ", xacro_path, " agent_name:=", agent_name, " use_ros_control:=", use_ros_control])
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

    ros_gz_bridge_ros_control = Node(
        package = "ros_gz_bridge",
        executable = "parameter_bridge",
        name = "ros_gz_bridge_ros_control",
        arguments = ["--ros-args", "-p", f"config_file:={bridge_path_1}"],
        condition = IfCondition(use_ros_control)
    )

    ros_gz_bridge_gazebo_control = Node(
        package = "ros_gz_bridge",
        executable = "parameter_bridge",
        name = "ros_gz_bridge_gazebo_control",
        arguments = ["--ros-args", "-p", f"config_file:={bridge_path_2}"],
        condition = UnlessCondition(use_ros_control)
    )

    diff_drive_spawner = Node(
        package = "controller_manager",
        executable = "spawner",
        condition = IfCondition(use_ros_control),
        arguments = ["diff_controller"],
    )

    joint_broadcaster_spawner = Node(
        package = "controller_manager",
        executable = "spawner",
        condition = IfCondition(use_ros_control),
        arguments = ["joint_broad"], 
    )

    laser_scan_matcher = Node(
        package = "rf2o_laser_odometry",
        executable = "rf2o_laser_odometry_node",
        name = "rf2o_laser_odometry",
        output = "screen",
        parameters = [{
            "laser_scan_topic" : "/scan",
            "odom_topic" : "/lidar_odom",
            "publish_tf" : False,
            "base_frame_id" : f"{agent_name}_base_link",
            "odom_frame_id" : "lidar_odom",
            "init_pose_from_topic" : ""}],
        arguments = ["--ros-args", "--log-level", "rf2o_laser_odometry:=error"],
        condition = IfCondition(use_ros_control)
    )


    return LaunchDescription([
        use_sim_time_arg,
        use_ros_control_arg,
        world_arg,
        rsp, 
        gazebo,
        spawner, 
        ros_gz_bridge_ros_control,
        ros_gz_bridge_gazebo_control,
        diff_drive_spawner,
        joint_broadcaster_spawner,
        laser_scan_matcher
    ])
    
