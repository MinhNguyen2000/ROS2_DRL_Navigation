from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory

# what does this file launch?
# - robot_state_publisher
# - joint_state_publisher
# - rviz

def generate_launch_description():
    # set the required paths:
    pkg_path = get_package_share_directory("agent_bringup")
    xacro_path = PathJoinSubstitution([pkg_path, "urdf", "agent.urdf.xacro"])
    rviz_config_path = PathJoinSubstitution([pkg_path, "config", "agent_rviz.rviz"])

    # define the launch arguments:
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value = "false",
        description = "Whether or not to use sim time, defaulting to false."
    )

    use_gui = LaunchConfiguration("use_gui")
    use_gui_arg = DeclareLaunchArgument(
        "use_gui",
        default_value = "false",
        description = "Whether or not to launch joint state publisher with the GUI, defaulting to false"
    )

    use_ros_control = LaunchConfiguration("use_ros_control")
    use_ros_control_arg = DeclareLaunchArgument(
        "use_ros_control",
        default_value = "false",
        description = "If true, control the agent using ros2_control, otherwise, use Gazebo plugins. Defaults to false"
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

    jsp = Node(
        package = "joint_state_publisher", 
        executable = "joint_state_publisher",
        name = "joint_state_publisher",
        namespace = agent_name,
        parameters = [{"use_sim_time" : use_sim_time}],
        condition = UnlessCondition(use_gui)
    )

    jsp_gui = Node(
        package = "joint_state_publisher_gui",
        executable = "joint_state_publisher_gui",
        name = "joint_state_publisher_gui",
        namespace = agent_name,
        parameters = [{"use_sim_time" : use_sim_time}],
        condition = IfCondition(use_gui)
    )

    rviz = Node(
        package = "rviz2",
        executable = "rviz2",
        name = "rviz2",
        namespace = agent_name, 
        output = "screen",
        arguments = ["-d", rviz_config_path],
        parameters = [{"use_sim_time" : use_sim_time}]
    )

    return LaunchDescription([
        use_sim_time_arg,
        use_gui_arg,
        use_ros_control_arg,
        rsp,
        jsp,
        jsp_gui,
        rviz
    ])



