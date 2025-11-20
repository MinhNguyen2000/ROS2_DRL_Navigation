# import the MakeEnv class written by Matt
# from environments import MakeEnv

from typing import Sequence
import numpy as np 
import gymnasium as gym
import mujoco as mj
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.registration import register

import os, json
from model_creation import MakeEnv

class Nav2D(MujocoEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 500}

    def __init__(self, 
                 json_file: str = "environment_params.json",
                 frame_skip: int = 2,
                 default_camera_config: dict[str, float | int] | None = None,
                 render_mode: str = "rgb_array",
                 width: int = 480,
                 height: int = 480,
                 camera_id: int | None = None,
                 camera_name: str | None = None,
                 reward_scale_options: dict[str, float] | None = None,
                 randomization_options: dict[str, float] | None = None,
                 visual_options: dict[int, bool] | None = None,
                 is_eval: bool = False
                ):
        ''' class constructor to initialize the environment (Mujoco model and data), the observation space, and renderer
        
        Arguments:
            json_file:              a string that contains the name of the environment parameters json file, which
                                    contains compiler info, visual settings, and element settings (ground, wall, light, 
                                    agent, goal)
            frame_skip:             number of frames skipped in the gymnasium MuJoCo renderer
            default_camera_config:  a dict object that contains camera placement information, such as 
                                    ``azimuth``, ``elevation``, ``distance``, ``lookat``
            render_mode:            a string that specifies the MuJoCo renderer mode, such as ``human``, ``rgb_array``, or ``None``
            width:                  width of the rendering window
            height:                 height of the rendering window
            camera_id:              id of the default camera in the rendering window
            camera_name:            name of the default camera in the rendering window
        '''

        """ Observation and Action Spaces """

        self.width, self.height = width, height
        self.render_mode = render_mode
        self.camera_name = camera_name
        self.camera_id = camera_id
        self.frame_skip = frame_skip
        self.n_rays = 8

        self.episode_counter = 0
        self.agent_frequency    = randomization_options.get("agent_freq", 1)    if randomization_options else 1
        self.goal_frequency     = randomization_options.get("goal_freq", 5)    if randomization_options else 5
        self.obstacle_frequency = randomization_options.get("obstacle_freq", 1) if randomization_options else 1

        self.agent_randomize = False
        self.goal_randomize = False
        self.obstacle_randomize = False

        # --- load the simulation parameters
        dir_path = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(dir_path,json_file)
        with open(json_path) as f:
            params = json.load(f)
        self.size = params["ground_settings"]["internal_length"]
        self.agent_radius = params["agent_settings"]["radius"]

        scaled_inner_length = 2 * (self.size - self.agent_radius)
        self.dmax = np.sqrt(2 * scaled_inner_length ** 2, dtype = np.float32)

        # # --- randomization bounds
        self.angle_bound = np.pi

        self.agent_bound_final = self.size - self.agent_radius
        self.agent_bound = self.agent_bound_final

        self.goal_bound_init = 0
        self.goal_bound_final = self.size - 2*self.agent_radius
        self.goal_bound = self.goal_bound_init
        
        # --- define the uninitialized location of the agent and the target
        # self._agent_loc = self.np_random.uniform(size=2, low=-self.agent_bound_init, high=self.agent_bound_init)
        # self._task_loc = self.np_random.uniform(size=2, low=-self.goal_bound_init, high=self.goal_bound_init)
        self._agent_loc = [0, 0]
        self._task_loc = [0, 0]
        
        # --- load simulation params and initialize the simulation
        env =  MakeEnv(params)
        env.make_env(agent_pos = self._agent_loc, 
                     task_pos = self._task_loc, 
                     n_rays = self.n_rays)
        self.model = env.model
        self.model.vis.global_.offwidth = width
        self.model.vis.global_.offheight = height
        self.data = mj.MjData(self.model)

        # --- define the observation space
        self._set_observation_space()
        
        #--- pre allocate observation array once
        self._obs_buffer = np.zeros(self._obs_space_size, dtype = np.float32)
        
        # --- define the action space
        self._set_action_space()

        self.init_qpos = self.data.qpos.ravel().copy()
        self.agent_init = np.zeros(3)
        self.init_qvel = self.data.qvel.ravel().copy()

        # --- initialize the renderer
        if "render_fps" in self.metadata:
            assert (
                int(np.round(1.0 / self.dt)) == self.metadata["render_fps"]
            ), f'Expected value: {int(np.round(1.0 / self.dt))}, Actual value: {self.metadata["render_fps"]}'

        # Delay creating the heavy renderer object until it's actually needed.
        # Avoid mutable default args by normalizing visual_options here.
        from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

        self._visual_options = {} if visual_options is None else visual_options
        self.mujoco_renderer = None
        
        if self.render_mode == "human":
            # Only instantiate the renderer if rendering was requested.
            self.mujoco_renderer = MujocoRenderer(
                self.model,
                self.data,
                default_cam_config = default_camera_config,
                width           = self.width,
                height          = self.height,
                max_geom        = 1000,
                camera_id       = self.camera_id,
                camera_name     = self.camera_name,
                visual_options  = self._visual_options,
            )

        self.window = None
        self.clock = None
        
        self.agent_id = self.model.body("agent").id
        self.goal_id = self.model.body("goal").id


        # --- termination conditions
        self.distance_threshold = self.agent_radius
        self.dist_progress_count = 0
        self.head_progress_count = 0
        self.progress_threshold = 200   # number of maximum allowable episodes where the agent has not made any progress toward the goal
        self.obstacle_threshold = 0.05 + self.agent_radius

        # --- scale of each reward
        self.rew_head_scale             = reward_scale_options.get("rew_head_scale", 1)             if reward_scale_options else 1
        self.rew_head_approach_scale    = reward_scale_options.get("rew_head_approach_scale", 100)  if reward_scale_options else 100
        self.rew_dist_scale             = reward_scale_options.get("rew_dist_scale", 1)             if reward_scale_options else 1
        self.rew_dist_approach_scale    = reward_scale_options.get("rew_dist_approach_scale", 100)  if reward_scale_options else 100
        self.rew_goal_scale             = reward_scale_options.get("rew_goal_scale", 5000)          if reward_scale_options else 5000
        self.rew_obst_scale             = reward_scale_options.get("rew_obst_scale", -1000)         if reward_scale_options else -1000
        self.rew_time                   = reward_scale_options.get("rew_time", -0.05)               if reward_scale_options else -0.05

        # --- intialize reward components
        self.rew_head_scaled = 0
        self.rew_head_approach_scaled = 0
        self.rew_dist_scaled = 0
        self.rew_dist_approach_scaled = 0

        # --- whether an evaluation environment or not
        self.is_eval = is_eval

    def _set_observation_space(self):
        ''' internal method to set the bounds on the observation space
        
        Order of the bounds
            (3, ): agents's x, y, z joint positions, where x and y are bounded by the arena size
            (3, ): agent's x, y, z joint velocities
            (2, ): goal's x + y body positions, where x and y are bounded by the arena size
            (n_rays, ): LiDAR scans'''
        # define the obs_space_size:
        self._obs_space_size = 3 + 3 + 2 + self.n_rays

        # set the scale on the observation space:
        # initialize the bounds as [-1, +1] scaled by some amount:
        low = -np.ones((self._obs_space_size,),dtype=np.float32) * self.dmax
        high = np.ones((self._obs_space_size,),dtype=np.float32) * self.dmax
        
        # set the x-y bounds of the agent and goal as half the arena size
        low[[0, 1, 6, 7]] = -(self.size - self.agent_radius)
        high[[0, 1, 6, 7]] = self.size - self.agent_radius

        # set the angular bounds:
        low[2] = 0.0
        high[2] = 2*np.pi

        self.observation_space = gym.spaces.Box(
            low=low,        # [x_min, y_min, target_x_min, target_y_min]
            high=high,        # [x_max, y_max, target_x_max, target_y_max]
            dtype=np.float32)
        return self.observation_space
    
    def _set_action_space(self):
        ''' internal method to set the bounds on the agent's local x_linear, y_linear and z_angular velocities'''
        # set the low and high of the action space:
        self.action_low = np.array([0, -1.0], dtype=np.float32)
        self.action_high = np.array([1.0, 1.0], dtype=np.float32)
        # self.action_low = np.array([-1.0], dtype=np.float32)
        # self.action_high = np.array([1.0], dtype=np.float32)

        self.action_space = gym.spaces.Box(low=self.action_low, high=self.action_high, dtype=np.float32)
        return self.action_space
    
    def _set_goal_bound(self, ratio: float):
        self.goal_bound = ratio * self.goal_bound_final
    
    def _get_obs(self):
        ''' internal method to obtain the location of agent/goal and the simulated LiDAR scan at any instance 
        
        Returns:
            tuple:
                a tuple containing the agent observation, goal observation, and LiDAR scan

                1. ``agent_obs``:  a stacked (6,) np.ndarray with the following components
                                    ``agent_pos`` (2,) - the global position of the agent (to compare to the global goal postion)
                                    ``agent_heading`` (1, ) - the local heading of the agent (w.r.t the agent's starting coordinate frame)
                                    ``agent_vel`` (3,) - the local velocities of the agent (w.r.t the agent's starting coordinate frame)
                
                2. ``goal_obs``:   a (3,) np.ndarray with the x-y-z position of the goal in the global frame

                3. ``lidar_obs``:  a np.ndarray with the LiDAR reading(s)
        '''

        #--- modify obs buffer inplace instead of concatenation overhead (time + memory)
        self._obs_buffer[0:2] = self.data.xpos[self.agent_id][:2]
        self._obs_buffer[2] = self.data.qpos[2]
        self._obs_buffer[3:6] = self.data.qvel[0:3]
        self._obs_buffer[6:8] = self.data.xpos[self.goal_id][:2]
        self._obs_buffer[8:] = self.data.sensordata[:-1]
        return self._obs_buffer
    
    def reset_model(self, 
                    agent_randomize: bool = False, 
                    goal_randomize: bool = False, 
                    obstacle_randomize: bool = False):
        noise_low = -0.1
        noise_high = 0.1

        # get a copy of the initial_qpos
        qpos = np.copy(self.init_qpos)      # initially agent is at [0,0], goal is at [-0.5, -0.5]
        qvel = np.copy(self.init_qvel)

        # if it is time to randomize the goal:
        if goal_randomize:
            # randomize the X,Y position of the goal by randomly sampling in a box around the center of the worldbody:
            qpos[3:5] = self.np_random.uniform(size=2, low=-self.goal_bound, high=self.goal_bound)
            self.init_qpos[3:5] = qpos[3:5]

        # if it is time to randomize the agent:
        if agent_randomize:
            # randomize the X,Y position of the agent by randomly sampling in a box around the center of the worldbody:
            qpos[0:2] = self.np_random.uniform(size=2, low=-self.agent_bound, high=self.agent_bound)

            # randomize the pose of the agent by randomly sampling between 0 and 2*pi:
            heading = self.get_heading(agent_pos=qpos[0:2], goal_pos=qpos[3:5])
            
            qpos[2] = self.np_random.uniform(size = 1, low = heading - self.angle_bound / 2, high = heading + self.angle_bound / 2)

            # randomize the velocity of the agent:
            qvel[0:2] = self.np_random.uniform(size=2, low=noise_low, high=noise_high)

            self.init_qpos[0:3] = qpos[0:3]
            self.agent_init = qpos[0:3]
            self.init_qvel[0:2] = qvel[0:2]

        if obstacle_randomize:
            pass

        self.set_state(qpos, qvel)
        ob = self._get_obs() 

        # get initial agent-goal distance
        agent_pos = ob[0:2]
        goal_pos = ob[6:8]
        self.d_goal_last = self._get_l2_distance(agent_pos, goal_pos)

        # get the last angular difference:
        required_heading = self.get_heading(agent_pos=agent_pos, goal_pos=goal_pos)
        self.prev_abs_diff = abs((required_heading - qpos[2] % (2*np.pi) + np.pi) % (2*np.pi) - np.pi)

        # reset the distance progress count
        self.dist_progress_count = 0
        self.head_progress_count = 0
        return ob
    
    def get_heading(self, 
                    agent_pos: list, 
                    goal_pos: list):
        # this function gets the heading based on an agent_pos and a goal_pos
        diff = goal_pos - agent_pos

        # heading:
        heading = np.arctan2(diff[1], diff[0], dtype = np.float32) % (2*np.pi)

        return heading

    def reset(self,
              seed: int | None = None,
              options: dict | None = None):
        
        # increment a counter:
        self.episode_counter += 1
        
        # call the reset method of the parent class:
        super().reset(seed = seed)

        # reset model data:
        mj.mj_resetData(self.model, self.data)
        
        # Log the information from this run before reset
        info = {"agent_init": self.agent_init,
                "rew_head": self.rew_head_scaled, 
                "rew_head_approach": self.rew_head_approach_scaled}

        # --- CHECK RANDOMIZATION CONDITIONS
        
        if not self.is_eval:
            # agent randomization
            if self.episode_counter == 1 or self.episode_counter % self.agent_frequency == 0:
                self.agent_randomize = True

            # goal randomization, with goal bound increase handled externally
            if self.episode_counter % self.goal_frequency == 0:
                self.goal_randomize = True

            # if self.episode_counter % self.obstacle_frequency == 0:
            #     self.obstacle_randomize = True

            # reset mujoco model:
            ob = self.reset_model(self.agent_randomize, self.goal_randomize, self.obstacle_randomize)
        
        else: # if an evaluation env, always reset and with the largest bound possible
            self.agent_bound = self.agent_bound_final
            self.goal_bound = self.goal_bound_final
            ob = self.reset_model(agent_randomize=True,
                                  goal_randomize=False,
                                  obstacle_randomize=True)

        # # reset flags:
        self.agent_randomize = False
        self.goal_randomize = False
        self.obstacle_randomize = False

        # form observation:
        ob = self.reset_model()

        # render if mode == "human":
        if self.render_mode == "human":
            self.render()
        return ob, info

    def _get_l2_distance(self, point_a: Sequence, point_b: Sequence):
        ''' internal method to obtain the Cartesian (l_2) distance between two points in 2D space
        
        Arguments:
            point_a:    a list or sequence containing at least the x-y coordinates of the first point
            point_b:    a list or sequence containing at least the x-y coordinates of the second point
        Returns:
            the 2-D Cartesian distance between the two points
        '''
        return np.linalg.norm(point_a[0:2]-point_b[0:2])

    def step(self, action):
        ''' method to execute one simulation step given the velocity command to the agent

        Arguments:
            action: the control action sent to the robot in terms of the local linear and angular velocities
        
        Returns:
            tuple:
                a tuple of (next observation, reward, termination)
                1. next_observation:    tuple containing the agent state, goal state, and LiDAR scan after simulating one step
                2. reward (float):      reward obtained after simulating the step
                3. term (bool):         whether the episode is terminated
        '''
        # 1. move the simulation forward with the TRANSFORMED action (w.r.t. original frame)
        # action_pre = np.array([0, 0, action[0]], dtype=np.float32)
        action_pre = np.array([action[0], 0, action[1]], dtype=np.float32)
        action_rot = np.copy(action_pre)

        # get angle:
        theta = self.data.qpos[2]   # theoretically faster than a function call

        # action transformed into global frame:
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        action_rot[0] = cos_theta * action_pre[0] - sin_theta * action_pre[1]
        action_rot[1] = sin_theta * action_pre[0] + cos_theta * action_pre[1]

        # scale the action as necessary:
        action_rot[0:2] *= 2
        action_rot[2] *= 3

        self.data.qvel[0:3] = action_rot

        # step the mujoco model:
        mj.mj_step(self.model, self.data, nstep=self.frame_skip)

        # 2. collect the new observation (LiDAR simulation, location of agent/goal using the custom _get_obs())\
        nobs = self._get_obs()
        # TODO - How can I make this more robust to future changes
        agent_pos = nobs[0:2]
        goal_pos = nobs[6:8]
        lidar_obs = nobs[8:]

        # use value of theta AFTER stepping:
        theta = nobs[2]

        # 3. termination condition: 
        # when the agent is close to the goal:
        d_goal = self._get_l2_distance(agent_pos, goal_pos)
        distance_cond = d_goal < self.distance_threshold

        # when the agent is close to obstacles:
        obstacle_cond = np.min(lidar_obs) < self.obstacle_threshold

        # get the difference in positions:
        required_heading = self.get_heading(agent_pos, goal_pos) 

        # wrap the current agent position between 0 and 2pi:
        wrapped_theta = theta % (2*np.pi)

        # find the absolute value of the difference in heading:
        abs_diff = np.abs((required_heading - wrapped_theta + np.pi) % (2 * np.pi) - np.pi)
        
        
        # when the agent has not reduced the d_goal for N steps, where N is 200
        if d_goal > self.d_goal_last: 
            self.dist_progress_count += 1
        else:
            self.dist_progress_count = 0

        if abs_diff >= self.prev_abs_diff:
            self.head_progress_count += 1
        else:
            self.head_progress_count = 0
        
        term = distance_cond or obstacle_cond or (self.dist_progress_count >= self.progress_threshold) or (self.head_progress_count >= self.progress_threshold)
        
        info = {}
        # 4. reward
        if distance_cond:
            rew = self.rew_goal_scale
        elif obstacle_cond:
            rew = self.rew_obst_scale
        else:
            #---  BASE HEADING REWARD:
            # penalize based on the absolute difference in heading:
            rew_head = 1.0 - np.tanh(3 * abs_diff / np.pi)

            # bonus reward for being within +/- 5 degree of the desired trajectory:
            angle_threshold = 2.5
            angle_threshold_rad = angle_threshold / 180 * np.pi
            if abs_diff <= angle_threshold_rad:
                rew_head += 1 - 1 / angle_threshold_rad * abs_diff

            # scale reward:
            self.rew_head_scaled = self.rew_head_scale * rew_head

            #--- HEADING APPROACH REWARD:
            rew_head_approach = max((self.prev_abs_diff - abs_diff), 0)
            self.rew_head_approach_scaled = self.rew_head_approach_scale * rew_head_approach

            #--- TIME REWARD:
            rew_time = -0.05    # going to keep this very small relative to the reward scale

            #--- DISTANCE REWARD:
            # this reward term incentivizes closing the distance between the agent and the goal:
            rew_dist = 1 - np.tanh(5 * d_goal / self.dmax)
            self.rew_dist_scaled = self.rew_dist_scale * rew_dist

            #--- DISTANCE APPROACH REWARD:
            rew_dist_approach = max((self.d_goal_last - d_goal), 0)
            self.rew_dist_approach_scaled = rew_dist_approach * self.rew_dist_approach_scale

            #--- TOTAL REWARD TERM:
            rew = self.rew_head_scaled + self.rew_head_approach_scaled + self.rew_dist_scaled + self.rew_dist_approach_scaled + rew_time

            # print to user:
            if self.render_mode == "human":
                print(f" @ episode {self.episode_counter} | vel: {action_rot.round(3)} | rew_head: {self.rew_head_scaled:.4f} | rew_head_approach: {self.rew_head_approach_scaled:.4f} | rew_dist: {self.rew_dist_scaled:.4f} | rew_dist_approach: {self.rew_dist_approach_scaled:.4f} | total: {rew:.5f}                                                                              ", end="\r")
            # info = {"rew_head": self.rew_head_scaled, "rew_head_approach" : self.rew_head_approach_scaled, "rew_dist_approach" : self.rew_dist_approach_scaled}

        # advance d_goal history:
        self.d_goal_last = d_goal
        self.prev_abs_diff = abs_diff
        
        # 5. info (optional):
        # info = {"reward": rew, "dist_cond": distance_cond, "obst_cond": obstacle_cond}
        # info = {}
        if term:
            info["is_success"] = bool(distance_cond)
        
        # 6. render if render_mode human:
        if self.render_mode == "human":
            self.render()

        return nobs, rew, term, False, info

register(
    id="Nav2D-v0",
    entry_point="nav2d:Nav2D",
    max_episode_steps=1_000,
)