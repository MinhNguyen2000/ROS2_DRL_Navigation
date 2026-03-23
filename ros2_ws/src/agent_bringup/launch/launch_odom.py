import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.actions import Node, PushRosNamespace
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory

# what does this file launch?
# - laser_scan_matcher
# - imu/lidar filter node
# - extended kalman filter

def generate_launch_description():
    # set the required paths:
    pkg_path = get_package_share_directory("agent_bringup")
    ekf_params_path = os.path.join(pkg_path, "config", "ekf_params.yaml")

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
    laser_scan_matcher = Node(
        package = "rf2o_laser_odometry",
        executable = "rf2o_laser_odometry_node",
        name = "rf2o_laser_odometry",
        namespace = agent_name,
        output = "screen",
        parameters = [{
            "laser_scan_topic" : "scan",
            "odom_topic" : "lidar_odom",
            "publish_tf" : False,
            "base_frame_id" : f"{agent_name}_base_link",
            "odom_frame_id" : "odom",
            "init_pose_from_topic" : "",
            "freq" : 60.0}],
    )

    covariance_filter_node = Node(
        package = "covariance_filter",
        executable = "covariance_filter_node",
        name = "covariance_filter",
        output = "screen",
    )

    covariance_filter_node = GroupAction(
        actions = [
            PushRosNamespace(agent_name),
            covariance_filter_node
        ]
    )

    ekf_node = Node(
        package = "robot_localization",
        executable = "ekf_node",
        name = "ekf_filter_node",
        namespace = agent_name, 
        output = "screen",
        parameters = [ekf_params_path, {"use_sim_time" : use_sim_time}],
        remappings = [
                    #   ("wheel_odom", f"{agent_name}/wheel_odom"),
                    #   ("lidar_odom_filtered", f"{agent_name}/lidar_odom_filtered"),
                    #   ("imu_data_filtered", f"{agent_name}/imu_data_filtered"),
                      ("odometry/filtered", "odom")],
    )

    return LaunchDescription([
        use_sim_time_arg, 
        laser_scan_matcher,
        covariance_filter_node,
        ekf_node
    ])