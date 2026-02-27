#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
import copy

class CovFilter(Node):
    def __init__(self):
        super().__init__("cov_filter")
        # print to terminal so I know that it is working:
        self.get_logger().info("Covariance filter node running")

        # topics that the node is subscribing to:
        self.imu_sub = self.create_subscription(
            Imu,
            "/imu_data", 
            self.imu_callback,
            10
        )

        self.lidar_sub = self.create_subscription(
            Odometry, 
            "/lidar_odom",
            self.lidar_callback,
            10
        )

        # topics that the node is publishing to:
        self.imu_pub = self.create_publisher(
            Imu,
            "/imu_data_filtered",
            10
        )

        self.lidar_pub = self.create_publisher(
            Odometry, 
            "/lidar_odom_filtered",
            10
        )

        # IMU covariances:
        # r, p, y row major
        self.imu_orientation_covariance = [1.0,
                                       0.0,
                                       0.0,
                                       0.0,
                                       1.0,
                                       0.0,
                                       0.0,
                                       0.0,
                                       0.05]
        
        self.imu_angular_velocity_covariance = [1.0,
                                            0.0,
                                            0.0,
                                            0.0,
                                            1.0,
                                            0.0,
                                            0.0,
                                            0.0,
                                            0.05]
        
        self.imu_linear_acceleration_covariance = [1.0,
                                               0.0,
                                               0.0,
                                               0.0,
                                               1.0,
                                               0.0,
                                               0.0,
                                               0.0,
                                               1.0]
        
        # LiDAR covariances:
        self.lidar_pose_covariance = [0.001,  0.0,   0.0, 0.0, 0.0, 0.0,
                                      0.0,    0.001, 0.0, 0.0, 0.0, 0.0,
                                      0.0,    0.0,   1.0, 0.0, 0.0, 0.0,
                                      0.0,    0.0,   0.0, 1.0, 0.0, 0.0,
                                      0.0,    0.0,   0.0, 0.0, 1.0, 0.0,
                                      0.0,    0.0,   0.0, 0.0, 0.0, 0.05]
          
        self.lidar_twist_covariance = [0.1, 0.0,  0.0,  0.0,  0.0,  0.0,
                                      0.0,  0.1,  0.0,  0.0,  0.0,  0.0,
                                      0.0,  0.0,  1.0,  0.0,  0.0,  0.0,
                                      0.0,  0.0,  0.0,  1.0,  0.0,  0.0,
                                      0.0,  0.0,  0.0,  0.0,  1.0,  0.0,
                                      0.0,  0.0,  0.0,  0.0,  0.0,  0.05]

    def imu_callback(self, msg):
        msg_fixed = copy.deepcopy(msg)
        msg_fixed.header.frame_id = "agent_imu_link"
        msg_fixed.orientation_covariance = self.imu_orientation_covariance
        msg_fixed.angular_velocity_covariance = self.imu_angular_velocity_covariance
        msg_fixed.linear_acceleration_covariance = self.imu_linear_acceleration_covariance
        self.imu_pub.publish(msg_fixed)

    def lidar_callback(self, msg):
        msg_fixed = copy.deepcopy(msg)
        msg_fixed.pose.covariance = self.lidar_pose_covariance
        msg_fixed.twist.covariance = self.lidar_twist_covariance
        self.lidar_pub.publish(msg_fixed)

def main(args=None):
    rclpy.init(args=args)
    node = CovFilter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()