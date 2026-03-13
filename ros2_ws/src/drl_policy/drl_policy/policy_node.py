import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

import numpy as np
# TODO - import torch related packages for DRL inference (torch and torch.nn)
from ament_index_python.packages import get_package_share_directory
import os

class PolicyNetwork(Node):
    '''TODO - Define the inference network architecture of the trained DRL actor network'''
    def __init__(self, obs_dim: int, action_dim: int):
        pass

    def forward(self, x):
        pass

class DRLPolicyNode(Node):
    def __init__(self):
        super().__init__('drl_policy_node')

        # TODO - declare and store parameters/runtime arguments (obs_dim, act_dim, min/max vel, control/inference rate)
        self.declare_parameter('inference_rate', 10.0)      # Hz

        # TODO - load the model

        # --- State storage ---
        self.latest_odom: Odometry | None = None
        self.latest_scan: LaserScan | None = None

        # --- Subscribers & Publisher ---
        self.odom_sub = self.create_subscription(
            Odometry,
            'odom',        # TODO - make the topic name dynamic according to /agent_name/odom namespace when this node is launched
            self.odom_callback,
            10
        )

        self.lidar_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.lidar_callback
        )

        # TODO - create a subscriber to listen to the locations of the goal (w.r.t agent odom frame). Need message type and topic information
        # self.goal_sub = self.create_subscription()

        self.cmd_pub = self.create_publisher(
            Twist,
            'cmd_vel',     # TODO - make the topic name dynamic according to /agent_name/cmd_vel namespace when this node is launched
            10
        )

        # --- Control inference frequency ---
        rate = self.get_parameter('inference_rate').value
        self.timer = self.create_timer(1.0 / rate, self.run_inference)

    def odom_callback(self, msg: Odometry):
        self.latest_odom = msg

    def lidar_callback(self, msg: LaserScan):
        self.latest_scan = msg

    def extract_obs(self, odom: Odometry, scan: LaserScan) -> np.ndarray:
        '''
        Build the observation from /odom messages to match the DRL policy's observation
        The observation space contains
        (dx, dy, dgoal, s_theta, c_theta, s_phi, c_phi, vx, vy, LiDAR scans)
        
        :param odom: odometry message from /odom (filtered wheel odometry, LiDAR matching, and IMU)
        :type odom: Odometry

        :return: Description
        :rtype: ndarray
        '''

        pos = odom.pose.pose.position           # for dx and dy
        ori = odom.pose.pose.orientation        # for theta
        lin_vel = odom.twist.twist.linear       # for vx and vy

        lidar_raw = scan.ranges

    def run_inference(self):

        if self.latest_odom is None:
            self.get_logger().warn('No odometry received yet - skipping inference.')
        elif self.latest_scan is None:
            self.get_logger().warn('No LiDAR scan received yet - skipping inference.')

        # --- 1. Extract observation ---
        obs = self.extract_obs(self.latest_odom, self.latest_scan)

        # --- 2. DRL policy inference => action ---

        # --- 3. Send the action command ---
        cmd = Twist()
        # cmd.linear.x = 
        # cmd.angular.z = 
        self.get_logger().debug(
            f"Linear vel: {cmd.linear.x: 5.3f} | Angular vel: {cmd.angular.z: 5.3f}"
        )
        self.cmd_pub.publish(cmd)
        pass