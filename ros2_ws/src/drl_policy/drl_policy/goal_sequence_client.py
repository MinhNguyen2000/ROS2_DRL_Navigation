import sys
import rclpy
from rclpy.node import Node
from rclpy.task import Future
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped

from drl_interfaces.action import NavigateToGoalSequence

PATHS: dict[str, list[tuple[float, float]]] = {
    "path_1": [(2.0, 1.0), (5.0, 1.0), (6.0, 6.0), (2.0, 5.0), [0.0, 0.0]],
    "path_2": [(2.0, 1.0), (5.0, 1.0), (6.0, 6.0), (2.0, 5.0), [0.0, 0.0]]
}

class GoalSequenceClient(Node):
    def __init__(self,
                 path_name: str, 
                 goal_tolerance: float, 
                 stop_on_failure: bool):
        super().__init__('goal_sequence_client')

        self._done = False

        # --- Check path name
        if path_name not in PATHS:
            self.get_logger().error(
                f"Unknown path '{path_name}'"
                f"Available paths: {list(PATHS.keys())}"
            )
            self._done = True
            return
        
        waypoints = PATHS[path_name]
        self.get_logger().info(
            f"Path '{path_name}' selected with {len(waypoints)} waypoints | "
            f"tolerance={goal_tolerance}m | stop_on_failure={stop_on_failure}"
        )

        self._client = ActionClient(
            self,
            NavigateToGoalSequence,
            'navigate_goal_sequence'
        )

        self.get_logger().info('Waiting for goal sequence server...')
        self._client.wait_for_server()

        # --- Create the goals ---
        goal = NavigateToGoalSequence.Goal()
        goal.waypoints = [self._make_pose(x, y) for x, y in waypoints]
        goal.goal_tolerance = goal_tolerance
        goal.stop_on_failure = stop_on_failure

        # --- Send goal ---
        self.get_logger().info("Sending path: '{path_name}'")
        self._send_future = self._client.send_goal_async(
            goal,
            feedback_callback=self._feedback_callback
        )
        self._send_future.add_done_callback(self._goal_accepted_callback)

    def _goal_accepted_callback(self, future: Future):
        self._goal_handle = future.result()
        if not self._goal_handle.accepted:
            self.get_logger().info('Sequence goal rejected by goal sequence server')
            self._done = True
            return
        
        self.get_logger().info('Sequence goal accepted')
        self._goal_handle.get_result_async().add_done_callback(self._result_callback)

    def _feedback_callback(self, feedback):
        f = feedback.feedback
        self.get_logger().info(
            f' Time: {f.elapsed_time: 5.2f} | '
            f'Waypoint {f.current_waypoint + 1}/{f.total_waypoints} | '
            f'current goal dist: {f.distance_to_current_goal: 5.3f}m'
        )

    def _result_callback(self, future: Future):
        '''
        Called at the end after the entire sequence is completed/failed
        '''
        result = future.result().result
        status = 'SUCCESS' if result.success else 'FAILED'
        self.get_logger().info(
            f'[{status}] {result.message} | '
            f'Completed {result.waypoints_completed} waypoint(s) | '
            f'Total distance: {result.total_distance: 5.3f}m'
        )
        self._done = True

    def cancel(self):
        self.get_logger().info('Sending cancel request to sequence server...')
        cancel_future = self._goal_handle.cancel_goal_async()

        while rclpy.ok() and not cancel_future.done():
            rclpy.spin_once(self, timeout_sec=0.1)

        if cancel_future.result():
            self.get_logger().info('Cancel acknowledge by the sequence server')
        else:
            self.get_logger().info('Cancel request rejectedd by the sequence server')

    def _make_pose(self, x: float, y: float):
        pose = PoseStamped()
        pose.header.stamp       = self.get_clock().now().to_msg()
        pose.header.frame_id    = 'odom'
        pose.pose.position.x    = x
        pose.pose.position.y    = y

        return pose
    
def main():
    rclpy.init()

    # Usage -> ros2 run drl_policy goal_sequence_client <path_name> <goal_tolerance> <stop_on_failure>
    # Example -> ros2 run drl_policy goal_sequence_client path_1 0.1 FALSE
    user_args = rclpy.utilities.remove_ros_args(sys.argv)[1:]
    path_name       = str(user_args[0])     if len(user_args) > 0 else "path_1"
    goal_tolerance  = float(user_args[1])   if len(user_args) > 1 else 0.1
    stop_on_failure = bool(user_args[2])    if len(user_args) > 2 else True

    node = GoalSequenceClient(path_name, goal_tolerance, stop_on_failure)

    try: 
        while rclpy.ok() and not node._done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.cancel()
        while rclpy.ok() and not node._done:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
