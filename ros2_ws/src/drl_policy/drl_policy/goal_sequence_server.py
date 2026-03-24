import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient, GoalResponse, CancelResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.action.client import ClientGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor       # to prevent blocking code while navigating to the goal
from rcl_interfaces.srv import GetParameters

from drl_interfaces.action import NavigateToGoal, NavigateToGoalSequence
from std_srvs.srv import Trigger
from geometry_msgs.msg import Quaternion

import os, json
import time
import numpy as np
import signal

class GoalSequenceServer(Node):
    '''
    Accepts a NavigateToGoalSequence action containing a list of waypoints and
    dispatches each waypoint sequentially to the drl_policy policy_node server
    using the NavigateToGoal action
    '''

    def __init__(self):
        super().__init__('goal_sequence_server')

        # this callback group allows the action client's callbacks to fire while
        # the execute_callback co-routine is awaiting
        self._callback_group = ReentrantCallbackGroup()

        self._sequence_server = ActionServer(
            self,
            NavigateToGoalSequence,
            'navigate_goal_sequence',
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            execute_callback=self._execute_callback,
            callback_group=self._callback_group
        )

        self._nav_client = ActionClient(
            self,
            NavigateToGoal,
            'navigate_to_goal',
            callback_group=self._callback_group
        )

        # Service client to grab model_name parameter from the drl_policy_server
        self._param_client = self.create_client(
            GetParameters,
            '/drl_policy_server/get_parameters'
        )
        
        if self._param_client.wait_for_service(timeout_sec=1.0):
            self._fetch_model_name()
        else:
            self.get_logger().warn('Could not reach policy_node to fetch model name')

        # Service server to save the trajectory from previous run
        self._save_srv = self.create_service(
            Trigger,
            'save_path',
            self._save_callback
        )

        self._pose_buffer = []
        self._elapsed_time = 0.0
        self._total_distance = 0.0

        self.get_logger().info('Goal sequence server ready')

        self.get_logger().info('Waiting for DRL policy server...')
        self._nav_client.wait_for_server()
        self.get_logger().info('Got it')
    
    def _goal_callback(self, goal_request: NavigateToGoalSequence.Goal):
        '''
        Activated whenever a goal is sent to this server, then decide whether
        to reject the goal (due to errors in the request) or accept
        '''
        n = len(goal_request.waypoints)
        self._path_name = goal_request.path_name

        if n == 0:
            self.get_logger().warn('Received empty goal lists - rejecting')
            return GoalResponse.REJECT
        self.get_logger().info(f'Accepting sequence of {n} goal(s)')
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle):
        self.get_logger().info('Goal sequence navigation cancel requested!')
        return CancelResponse.ACCEPT
    
    async def _execute_callback(self, seq_goal_handle: ServerGoalHandle):
        '''
        Iterate through the list of waypoints, send a NavigateToGoal action
        to the DRL policy node action server for each waypoint and wait for 
        the result
        '''

        self._pose_buffer = []  # clear the pose trajectory

        request         = seq_goal_handle.request
        waypoints       = request.waypoints
        tolerance       = request.goal_tolerance
        stop_on_fail    = request.stop_on_failure
        total           = len(waypoints)

        feedback_msg    = NavigateToGoalSequence.Feedback()
        feedback_msg.total_waypoints = total
        result_msg      = NavigateToGoalSequence.Result()
        result_msg.total_distance = 0.0
        result_msg.waypoints_completed = 0
        self._total_distance = 0.0

        start = time.time()

        # --- Check action server availability ---
        if not self._nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('navigate_to_goal action server not available')
            seq_goal_handle.abort()
            result_msg.success = False
            result_msg.message = 'DRL policy server unavailable'

        # --- Iterate through waypoints and send to action server ---
        for idx, waypoint in enumerate(waypoints):
            # --- Check for cancellation on the goal handle ---
            if seq_goal_handle.is_cancel_requested:
                seq_goal_handle.canceled()
                result_msg.success = False
                result_msg.message = f'Cancelled before reaching waypoint {idx}'
                return result_msg

            self.get_logger().info(
                f'Navigating to waypoint {idx + 1: 2d}/{total: 2d}: '
                f'({waypoint.pose.position.x:5.2f}, {waypoint.pose.position.y:5.2f})'
            )

            # --- Create the single-goal request ---
            nav_goal = NavigateToGoal.Goal()
            nav_goal.target_pose = waypoint
            nav_goal.goal_tolerance = tolerance

            send_goal_future = await self._nav_client.send_goal_async(
                nav_goal,
                feedback_callback=lambda fb, i=idx: self._relay_feedback(fb, seq_goal_handle, i, total, feedback_msg, start)
            )

            nav_goal_handle: ClientGoalHandle = send_goal_future

            # --- Check for goal acceptance ---
            if not nav_goal_handle.accepted:
                self.get_logger().warn(f'Waypoint {idx + 1} rejected by the policy node')

                if stop_on_fail:
                    seq_goal_handle.abort()
                    result_msg.success = False
                    result_msg.message = f'Waypoint {idx + 1} rejected'
                    return result_msg
                continue    # skip to the next waypoint

            # --- Wait for the current goal to finish ---
            result_future   = await nav_goal_handle.get_result_async()
            nav_result      = result_future.result

            result_msg.total_distance += nav_result.total_distance
            self._total_distance = result_msg.total_distance

            # --- Check for navigation success (1 goal) --- 
            if nav_result.success:
                result_msg.waypoints_completed += 1
                self.get_logger().info(f'Waypoint {idx + 1}/{total} reached.')
            else:
                self.get_logger().warn(f'Waypoint {idx + 1}/{total} failed: {nav_result.message}')
                if stop_on_fail:
                    seq_goal_handle.abort()
                    result_msg.success = False
                    result_msg.message = f'Stopped at waypoint {idx + 1}: {nav_result.message}'
                    return result_msg
                
        # --- Check if all waypoints done (all goals) ---
        all_waypoints = result_msg.waypoints_completed == total
        self._elapsed_time = time.time() - start
        seq_goal_handle.succeed()
        result_msg.success = all_waypoints
        result_msg.message = 'All waypoints completed' if all_waypoints else f'Completed {result_msg.waypoints_completed}/{total} waypoints.'
        return result_msg

    def _relay_feedback(self, 
                        nav_feedback_handle,
                        seq_goal_handle: ServerGoalHandle,
                        current_idx: int,
                        total: int,
                        feedback_msg: NavigateToGoalSequence.Feedback,
                        start: float
    ):
        '''
        Forward feedback from the DRL policy action server to the sequence
        caller
        '''
        
        feedback_msg.current_waypoint = current_idx
        feedback_msg.total_waypoints = total
        feedback_msg.elapsed_time = float(time.time() - start)

        # Extract from the policy node's NavigateToGoal action feedback
        fb = nav_feedback_handle.feedback
        feedback_msg.distance_to_current_goal = fb.distance_to_goal
        feedback_msg.current_pose = fb.current_pose

        seq_goal_handle.publish_feedback(feedback_msg)

        self._pose_buffer.append({
            "x": round(fb.current_pose.position.x, 3),
            "y": round(fb.current_pose.position.y, 3),
            "yaw": round(self._yaw_from_quaternion(fb.current_pose.orientation), 3)
        })  
    
    def _fetch_model_name(self):
        request = GetParameters.Request()
        request.names = ['model_name']
        future = self._param_client.call_async(request)
        future.add_done_callback(self._model_name_callback)

    def _model_name_callback(self, future):
        try:
            response = future.result()
            self._model_name = response.values[0].string_value
            self.get_logger().info(f'Navigating using model {self._model_name}')
        except Exception as e:
            self.get_logger().error(f'Failed to fetch model name: {e}')

    def _save_callback(self, request: Trigger.Request, response: Trigger.Response):
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        record_path_name = f"{self._model_name}_{self._path_name}.json"
        save_path = os.path.join(pkg_dir, '..', 'recorded_paths', record_path_name)

        data = {
            "model_name":       self._model_name,
            "path_name":        self._path_name,
            "elapsed_time":     self._elapsed_time,
            "total_distance":   self._total_distance,
            "poses":            self._pose_buffer,
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        response.success = True
        response.message = f'Save {len(self._pose_buffer)} poses to {save_path}'
        self.get_logger().info(response.message)
        return response
    
    def _yaw_from_quaternion(self, q: Quaternion) -> float:
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y**2 + q.z**2)
        return float(np.atan2(siny_cosp, cosy_cosp, dtype=np.float32))

def main():
    rclpy.init()
    node = GoalSequenceServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    # catch sigterm from GUI:
    signal.signal(signal.SIGTERM, lambda *args: executor.shutdown())

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
