# import the MakeEnv class written by Matt
# from environments import MakeEnv

from typing import Sequence
import numpy as np 
import gymnasium as gym
import mujoco as mj
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.registration import registry, register

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
            reward_scale_options:   a dictionary containing the value for each of the reward scale (dist, dist_app, head, head_app, and time)
            randomization_options:  a dictionary containing the randomization frequency of the agent, goal, and obstacles
            visual_options:         arguments to pass to the MuJoCo renderer to enable/disable visual elements (actuator axes, LiDAR rays)
            is_eval:                whether the environment is in evaluation mode, which alters the randomization scheme
        '''

        """ Observation and Action Spaces """

        self.width, self.height = width, height
        self.render_mode = render_mode
        self.camera_name = camera_name
        self.camera_id = camera_id
        self.frame_skip = frame_skip
        self.n_rays = 8

        self.linear_scale = 2
        self.angular_scale = 3

        # --- load the simulation parameters
        dir_path = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(dir_path,json_file)
        with open(json_path) as f:
            params = json.load(f)
        self.size = params["ground_settings"]["internal_length"]
        self.agent_radius = params["agent_settings"]["radius"]

        scaled_inner_length = 2 * (self.size - self.agent_radius)
        self.dmax = np.sqrt(2 * scaled_inner_length ** 2, dtype = np.float32)
        
        # --- define the uninitialized location of the agent and the target
        self._agent_loc = [-(self.size - 2*self.agent_radius), -(self.size - 2*self.agent_radius)]
        self._task_loc = [(self.size - 2*self.agent_radius), (self.size - 2*self.agent_radius)]
        
        # --- load simulation params and initialize the simulation
        env =  MakeEnv(params)
        env.make_env(agent_pos = self._agent_loc, 
                     task_pos = self._task_loc, 
                     n_rays = self.n_rays)
        self.model = env.model
        self.model.vis.global_.offwidth = width
        self.model.vis.global_.offheight = height
        self.data = mj.MjData(self.model)

        # --- OBSERVATION SPACE INITIALIZATION
        self._set_observation_space()
        
        #--- pre allocate observation array once
        self._obs_buffer = np.zeros(self._obs_space_size, dtype = np.float32)
        
        # --- ACTION SPACE INITIALIZATION
        self._set_action_space()

        self.init_qpos = self.data.qpos.ravel().copy()
        self.agent_init = np.zeros(3)
        self.init_qvel = self.data.qvel.ravel().copy()

        # --- RENDERER INITIALIZATION
        if "render_fps" in self.metadata:
            assert (
                int(np.round(1.0 / self.dt)) == self.metadata["render_fps"]
            ), f'Expected value: {int(np.round(1.0 / self.dt))}, Actual value: {self.metadata["render_fps"]}'

        # delay creating the heavy renderer object until it's actually needed.
        # avoid mutable default args by normalizing visual_options here.
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

        # --- RANDOMIZATION INITIALIZATION
        self.episode_counter = 0
        self.agent_frequency    = randomization_options.get("agent_freq", 1)    if randomization_options else 1
        self.goal_frequency     = randomization_options.get("goal_freq", 1)    if randomization_options else 1
        self.obstacle_frequency = randomization_options.get("obstacle_freq", 1) if randomization_options else 1

        self.agent_randomize = False
        self.goal_randomize = False
        self.obstacle_randomize = False

        # agent randomization bounds
        self.agent_bound_final = self.size - self.agent_radius
        self.agent_bound = self.agent_bound_final
        self.angle_bound = np.pi

        # goal randomization bounds
        self.goal_bound_init = 0
        self.goal_bound_final = self.size - 2*self.agent_radius
        self.goal_bound = self.goal_bound_final

        # --- TERMINATION CONDITION INITILIZATION
        self.distance_threshold = self.agent_radius
        self.dist_progress_count = 0
        self.head_progress_count = 0
        self.progress_threshold = 100   # number of maximum allowable episodes where the agent has not made any progress toward the goal
        self.obstacle_threshold = 0.05 + self.agent_radius

        # --- REWARD SCALE INITIALIZATION
        self.rew_dist_scale             = reward_scale_options.get("rew_dist_scale", 1)             if reward_scale_options else 1
        self.rew_dist_approach_scale    = reward_scale_options.get("rew_dist_approach_scale", 250)  if reward_scale_options else 100
        self.rew_head_scale             = reward_scale_options.get("rew_head_scale", 1)             if reward_scale_options else 1
        self.rew_head_approach_scale    = reward_scale_options.get("rew_head_approach_scale", 250)  if reward_scale_options else 100
        self.rew_goal_scale             = reward_scale_options.get("rew_goal_scale", 5000)          if reward_scale_options else 5000
        self.rew_obst_scale             = reward_scale_options.get("rew_obst_scale", -1000)         if reward_scale_options else -1000
        self.rew_time                   = reward_scale_options.get("rew_time", -0.25)               if reward_scale_options else -0.25

        # intialize the scaled reward components
        self.rew_head_scaled = 0
        self.rew_head_approach_scaled = 0
        self.rew_dist_scaled = 0
        self.rew_dist_approach_scaled = 0

        # --- whether an evaluation environment or not
        self.is_eval = is_eval

    def _set_observation_space(self):
        ''' 
        internal method to set the bounds on the observation space

        order of the bounds:
            (2, ): the dx and dy components of how far the agent is from the goal
            (2, ): the cos(theta) and sin(theta) components of the agent's heading
            (2, ): the cos() and sin() componets of the relative bearing
            (n_rays, ): LiDAR scans

        '''
        # define the obs_space_size:
        self._obs_space_size = 2 + 2 + 2 + self.n_rays

        # initialize the scale on the observation space as being between [-1, 1]:
        low  = -np.ones((self._obs_space_size, ), dtype = np.float32) 
        high =  np.ones((self._obs_space_size, ), dtype = np.float32)

        # dx and dy bounds
        low[0:2]  = -2 * (self.size - self.agent_radius)
        high[0:2] =  2 * (self.size - self.agent_radius)

        # LiDAR bounds
        low[6:]  = 0.0
        high[6:] = self.dmax + np.sqrt(2 * self.agent_radius**2)

        # set the observation space:
        self.observation_space = gym.spaces.Box(
            low = low,            
            high = high,         
            dtype = np.float32)
        
        return self.observation_space
    
    def _set_action_space(self):
        ''' internal method to set the bounds on the agent's local x_linear and z_angular velocities'''
        # set the low and high of the action space:
        self.action_low = np.array([0, -1.0], dtype = np.float32)
        self.action_high = np.array([1.0, 1.0], dtype = np.float32)

        self.action_space = gym.spaces.Box(low = self.action_low, high = self.action_high, dtype = np.float32)
        return self.action_space
    
    def _set_goal_bound(self, ratio: float):
        ''' internal method to control the active goal randomization boundary
        * this method is used for curriculum learnnng in the training script'''
        self.goal_bound = ratio * (self.goal_bound_final - self.goal_bound_init) + self.goal_bound_init
    
    def _set_agent_bound(self, ratio: float):
        ''' internal method to control the active goal randomization boundary
        * this method is used for curriculum learnnng in the training script'''
        self.agent_bound = ratio * (self.agent_bound_final - self.agent_bound_init) + self.agent_bound_init

    def _get_obs(self):
        ''' internal method to obtain the location of agent/goal and the simulated LiDAR scan at any instance 
        
        Returns: obs_buffer containing
            (2, ): the dx and dy components of how far the agent is from the goal
            (2, ): the cos(theta) and sin(theta) components of the agent's heading
            (n_rays, ): normalized LiDAR scans
        '''
        # need to compute the dx and dy between the agent and the goal:
        dx, dy = self.data.xpos[self.goal_id][:2] - self.data.xpos[self.agent_id][:2]

        # grab the heading of the agent:
        theta = self.data.qpos[2]
        c_theta = np.cos(theta, dtype=np.float32)
        s_theta = np.sin(theta, dtype=np.float32)

        # calculate the relative bearing from the agent to the goal
        bearing = np.arctan2(dy,        dx,         dtype = np.float32) % (2 * np.pi)
        heading = np.arctan2(s_theta,   c_theta,    dtype = np.float32) % (2 * np.pi)
        rel_bearing = abs((bearing - heading + np.pi) % (2*np.pi) - np.pi)
        c_bearing = np.cos(rel_bearing, dtype=np.float32)
        s_bearing = np.sin(rel_bearing, dtype=np.float32) 

        #--- modify obs buffer inplace instead of concatenation overhead (time + memory):
        self._obs_buffer[0:2] = dx, dy
        self._obs_buffer[2:4] = c_theta, s_theta
        self._obs_buffer[4:6] = c_bearing, s_bearing
        self._obs_buffer[6:]  = self.data.sensordata[:-1]

        return self._obs_buffer

    def reset_model(self, 
                    agent_randomize: bool = False, 
                    goal_randomize: bool = False, 
                    obstacle_randomize: bool = False):
        noise_low = -0.1
        noise_high = 0.1

        # get a copy of the initial_qpos
        qpos = np.copy(self.init_qpos)
        qvel = np.copy(self.init_qvel)

        # if it is time to randomize the goal:
        if goal_randomize:
            # randomize the X,Y position of the goal by randomly sampling in a box around the center of the worldbody:
            qpos[3:5] = self.np_random.uniform(size = 2, low = -self.goal_bound, high = self.goal_bound) - self._task_loc
            self.init_qpos[3:5] = qpos[3:5]

        # if it is time to randomize the agent:
        if agent_randomize:
            # randomize the X,Y position of the agent by randomly sampling in a box around the center of the worldbody:
            qpos[0:2] = self.np_random.uniform(size = 2, low = -self.agent_bound, high = self.agent_bound) - self._agent_loc

            # randomize the pose of the agent by randomly sampling within self.angle_bound/2 away from the required heading
            dx, dy = (qpos[3:5] + self ._task_loc) - (qpos[0:2] + self._agent_loc)
            bearing = np.arctan2(dy, dx, dtype=np.float32) % (2*np.pi)
            qpos[2] = self.np_random.uniform(size = 1, low = bearing - self.angle_bound / 2, high = bearing + self.angle_bound / 2) % (2 * np.pi)

            # randomize the velocity of the agent:
            qvel[0:2] = self.np_random.uniform(size = 2, low = noise_low, high = noise_high)

            self.init_qpos[0:3] = qpos[0:3]
            self.agent_init = qpos[0:3]
            self.init_qvel[0:2] = qvel[0:2]

        if obstacle_randomize:
            pass

        self.set_state(qpos, qvel)
        ob = self._get_obs()

        # get the previous d_goal:
        dx, dy = ob[0:2]
        self.d_goal_last = np.sqrt(dx**2  + dy**2)      # to track distance approach progress
        self.d_init = self.d_goal_last                  # to track overall distance progress
   
        # get the initial abs_diff
        c_bearing, s_bearing = ob[4:6]
        self.prev_abs_diff = np.arctan2(s_bearing, c_bearing, dtype=np.float32)

        # reset the distance progress count
        self.dist_progress_count = 0
        self.head_progress_count = 0
        return ob

    def reset(self,
              seed: int | None = None,
              options: dict | None = None):
        
        # increment episode counter:
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
            self.goal_bound = 0.3 * self.goal_bound_final
            ob = self.reset_model(agent_randomize = True,
                                  goal_randomize = True,
                                  obstacle_randomize = True)

        # reset flags:
        self.agent_randomize = False
        self.goal_randomize = False
        self.obstacle_randomize = False

        # render if mode == "human":
        if self.render_mode == "human":
            self.render()
        return ob, info

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
        action_pre = np.array([action[0], 0, action[1]], dtype = np.float32)
        action_rot = np.copy(action_pre)

        # get angle:
        theta = self.data.qpos[2]

        # action transformed into global frame:
        cos_theta = np.cos(theta, dtype = np.float32)
        sin_theta = np.sin(theta, dtype = np.float32)
        action_rot[0] = cos_theta * action_pre[0] - sin_theta * action_pre[1]
        action_rot[1] = sin_theta * action_pre[0] + cos_theta * action_pre[1]

        # scale the action as necessary:
        action_rot[0:2] *= self.linear_scale
        action_rot[2]   *= self.angular_scale

        self.data.qvel[0:3] = action_rot

        # step the mujoco model:
        mj.mj_step(self.model, self.data, nstep = self.frame_skip)

        # 2. collect the new observation (LiDAR simulation, location of agent/goal using the custom _get_obs())
        nobs = self._get_obs()      # this is [d_goal, abs_diff, LiDAR]
        dx, dy                  = nobs[0:2]
        c_theta, s_theta        = nobs[2:4]
        c_bearing, s_bearing    = nobs[4:6]
        lidar_obs               = nobs[6:]

        # get the cartesian distance between the agent and the goal:
        d_goal = np.sqrt(dx**2  + dy**2)

        # get the difference between the agents current heading, and the required heading:
        heading  = np.arctan2(s_theta, c_theta) % (2 * np.pi)
        abs_diff = np.arctan2(s_bearing, c_bearing)

        # 3. termination conditions: 
        # when the agent is close to the goal:
        distance_cond = d_goal < self.distance_threshold

        # when the agent is close to obstacles:
        obstacle_cond = np.min(lidar_obs) < self.obstacle_threshold
        
        # when the agent has not reduced the d_goal for N steps, where N is 200:
        if d_goal > self.d_goal_last: 
            self.dist_progress_count += 1
        else:
            self.dist_progress_count = 0

        # when the abs_diff is more than 15 degress and still growing:
        if abs_diff >= self.prev_abs_diff and (abs_diff > (15 / 180 * np.pi)):
            self.head_progress_count += 1
        else:
            self.head_progress_count = 0
        
        term = distance_cond or obstacle_cond or (self.dist_progress_count >= self.progress_threshold) or (self.head_progress_count >= self.progress_threshold)
        
        info = {}

        if term:
            info["is_success"]          = bool(distance_cond)
            info["obstacle_cond"]       = bool(obstacle_cond)
            info["dist_progress_cond"]  = (self.dist_progress_count >= self.progress_threshold)
            info["head_progress_cond"]  = (self.head_progress_count >= self.progress_threshold)
            
        # 4. reward:
        if distance_cond:
            rew = self.rew_goal_scale
        elif obstacle_cond:
            rew = self.rew_obst_scale
        else:
            # --- APPROACH REWARD:
            rew_dist_approach = self.d_goal_last - d_goal
            self.rew_dist_approach_scaled = self.rew_dist_approach_scale * rew_dist_approach

            # --- ALIGNMENT REWARD:
            rew_head = 1 + np.cos(abs_diff)
            self.rew_head_scaled = self.rew_head_scale * rew_head

            # --- TOTAL REWARD:
            rew = self.rew_dist_approach_scaled + self.rew_head_scaled + self.rew_time

            # --- PRINT TO USER:
            if self.render_mode == "human":
                print(f" @ episode {self.episode_counter} | rew_dist_approach: {self.rew_dist_approach_scaled:.4f} | rew_head: {self.rew_head_scaled:.4f} | total: {rew:.5f}                              ", end = "\r")

        # advance histories:
        self.d_goal_last = d_goal
        self.prev_abs_diff = abs_diff
        
        # 5. info (optional):
        # info = {"reward": rew, "dist_cond": distance_cond, "obst_cond": obstacle_cond}
        
        # 6. render if render_mode human:
        if self.render_mode == "human":
            self.render()

        return nobs, rew, term, False, info

ENV_ID = "Nav2D-v0"

if ENV_ID not in gym.envs.registry:
    register(
        id=ENV_ID,
        entry_point="nav2d:Nav2D",
        max_episode_steps=1_000,
    )