import sys
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.task import Future
from drl_interfaces.action import NavigateToGoal
from geometry_msgs.msg import PoseStamped

class GoalClient(Node):
    def __init__(self, x: float, y: float, goal_tolerance: float):
        super().__init__('goal_client')
        self._done = False

        self._client = ActionClient(
            self, 
            NavigateToGoal, 
            'navigate_to_goal'
        )

        self.get_logger().info('Waiting for action server...')
        self._client.wait_for_server()

        goal = NavigateToGoal.Goal()
        goal.target_pose = PoseStamped()
        goal.target_pose.header.stamp       = self.get_clock().now().to_msg()
        goal.target_pose.header.frame_id    = 'odom'
        goal.target_pose.pose.position.x    = x
        goal.target_pose.pose.position.y    = y
        goal.goal_tolerance                 = goal_tolerance

        self.get_logger().info(f'Sending goal: ({x: 5.3f},{y: 5.3f})')
        self._send_future = self._client.send_goal_async(
            goal,
            feedback_callback=self.feedback_callback
        )
        self._send_future.add_done_callback(self.goal_accepted_callback)

    def goal_accepted_callback(self, future: Future):
        '''Actions performed on client side when the server receives the goal'''
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return
        
        self.get_logger().info('Goal accepted')
        goal_handle.get_result_async().add_done_callback(self.result_callback)

    def feedback_callback(self, feedback):
        '''Acknowledging methods whenever the server send a feedback'''
        f = feedback.feedback
        self.get_logger().info(
            f'Time: {f.elapsed_time: 5.2f} | ' 
            f'Distance to goal: {f.distance_to_goal: 5.3f} m'
        )

    def result_callback(self, future):
        result = future.result().result
        self.get_logger().info(
            f'{result.message} | Total distance travelled: {result.total_distance: 5.3f}')
        self._done = True


def main():
    rclpy.init()

    # Usage -> ros2 run drl_policy goal_client 3.0 2.0 0.5
    user_args = rclpy.utilities.remove_ros_args(sys.argv)[1:]
    x = float(user_args[0]) if len(user_args) > 0 else 0.0
    y = float(user_args[1]) if len(user_args) > 1 else 0.0
    goal_tolerance = float(user_args[2]) if len(user_args) > 2 else 0.2

    node = GoalClient(x, y, goal_tolerance)
    while rclpy.ok() and not node._done:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()