# import the MakeEnv class written by Matt
# from environments import MakeEnv

from typing import Sequence
import numpy as np 
import gymnasium as gym
import mujoco as mj
from gymnasium.envs.mujoco import MujocoEnv

class Nav2D(MujocoEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 500}

    def __init__(self, 
                 xml_file: str = 'assets/env_test_minh.xml',
                 frame_skip: int = 2,
                 size: float = 5.0,
                 default_camera_config: dict[str, float | int] | None = None,
                 **kwargs
                #  render_mode: str | None = None,
                #  width: int = 480,
                #  height: int = 480,
                #  camera_id: int | None = None,
                #  camera_name: str | None = None
                ):
        
        """ Observation and Action Spaces """
        self.size = size       # assume 5x5 m^2 environment
        
        # --- define the uninitialized location of the agent and the target - to be set randomly in the reset() method
        self._agent_loc = np.array([0, 0], dtype=np.float32)
        self._target_loc = np.array([0, 0], dtype=np.float32)

        # --- define the observation space
        observation_space = gym.spaces.Box(
            low=-size/2,        # [x_min, y_min, target_x_min, target_y_min]
            high=size/2,        # [x_max, y_max, target_x_max, target_y_max]
            shape=(4,), dtype=np.float32)
        
        # --- initialize the simulator (model + data), set action space, and create a renderer
        MujocoEnv.__init__(
            self,
            model_path=xml_file,
            frame_skip=frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config,
            **kwargs
        )

        self.window = None
        self.clock = None

        self.rot_matrix = np.zeros((2,2))
        self.agent_id = self.model.body("agent").id
        self.goal_id = self.model.body("goal").id

        self.distance_threshold = 0.01

    # TODO - create a _get_obs() and _get_info() methods
    def _get_obs(self):
        ''' Function to obtain the LiDAR simulation scan and location of agent/goal at any instance '''
        
        # Grab the current pose of the robot
        agent_pos= self.data.xpos[self.agent_id][:2]                                    
        agent_heading = np.array(self.data.qpos[2], dtype = np.float32).reshape(1)     
        agent_obs = np.concatenate([
            agent_pos,      # Global coordinates (x_world, y_world)
            agent_heading,  # Local coordinate heading
            self.data.qvel  # Local velocities w.r.t the joints
            ]).ravel()

        # Grab the current location of the goal
        goal_obs = self.data.xpos[self.goal_id]

        # Grab the lidar sensor values
        lidar_obs = self.data.sensordata

        return agent_obs, goal_obs, lidar_obs
    
    # TODO - create the reset() method
    def reset():
        pass

    def _get_l2_distance(self, point_a: Sequence, point_b: Sequence):
        return np.sqrt(np.square(point_a[0]-point_b[0]) + np.square(point_a[1]-point_b[1]))

    # TODO - create the step() method
    def step(self, action):
        # 1. move the simulation forward with the TRANSFORMED action (w.r.t. original frame)
        agent_obs = self._get_obs()[0]
        theta = agent_obs[2]
        self.rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]], dtype=np.float32)
        action[:2] = self.rot_matrix @ action[:2]

        self.data.qvel[:] = action

        mj.mj_step(self.model, self.data, nstep=self.frame_skip)

        # 2. collect the new observation (LiDAR simulation, location of agent/goal using the custom _get_obs())\
        nobs = self._get_obs()
        agent_obs, goal_obs, lidar_obs = nobs

        # 3. termination condition 
        # when the agent is close to the goal
        d_goal = self._get_l2_distance(agent_obs[0:3], goal_obs)
        distance_cond = d_goal < self.distance_threshold
        # when the agent is close to obstacles
        obstacle_cond = False       # TODO - placeholder for when LiDAR observation is available
        term = distance_cond or obstacle_cond
        
        # 4. reward - TODO - placeholder for reward values
        if distance_cond:
            rew = 200
        elif obstacle_cond:
            rew = -100
        else:
            rew = 5*d_goal

        # 5. info (optional)

        # 6. render if render_mode human 
        # TODO - test the gymnasium mujoco render
        if self.render_mode == "human":
            self.render()

        return nobs, rew, term
    
    # TODO - create the render() method
    # TODO - create the close() method