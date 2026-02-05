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
                 default_camera_config: dict[str, float | int] = {"azimuth" : 90.0, "elevation" : -90.0, "distance" : 3.0, "lookat" : [0.0, 0.0, 0.0]},
                 render_mode: str = "rgb_array",
                 width: int = 480,
                 height: int = 480,
                 camera_id: int | None = None,
                 camera_name: str | None = None,
                 reward_scale_options: dict[str, float] | None = None,
                 randomization_options: dict[str, float] | None = None,
                 obstacle_options: dict[str, int] = {"n_obstacles": 0},
                 visual_options: dict[int, bool] | None = None,
                 n_ray_groups: int = 16,
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
        self.n_rays = 360
        self.n_ray_groups = n_ray_groups
        self._ray_per_group = int(self.n_rays/self.n_ray_groups)

        self.linear_scale = 2
        self.angular_scale = 3

        # --- load the simulation parameters
        dir_path = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(dir_path,json_file)
        with open(json_path) as f:
            params = json.load(f)
        self.size = params["ground_settings"]["internal_length"]
        self.agent_radius = params["agent_footprint_settings"]["radius"]
        self.goal_radius = params["task_settings"]["radius"]
        self.allowance = params["obstacle_settings"]["allowance"]

        scaled_inner_length = 2 * (self.size - self.agent_radius)
        self.dmax = np.sqrt(2 * scaled_inner_length ** 2, dtype = np.float32)

        # --- TERMINATION CONDITION INITILIZATION
        self.distance_threshold = self.agent_radius
        self.dist_progress_count = 0
        self.head_progress_count = 0
        self.progress_threshold = 100 if not is_eval else 500   # number of maximum allowable episodes where the agent has not made any progress toward the goal
        self.obstacle_threshold = self.allowance + self.agent_radius    # this is basically the footprint radius of the agent
        self.collision = False

        # --- RANDOMIZATION INITIALIZATION
        self.episode_counter = 0
        self.randomization_options = randomization_options
        self.randomization_freq    = randomization_options.get("randomization_freq", 1)    if randomization_options else 1

        self.reset_randomize = False

        # agent randomization bounds
        self.agent_bound_init  = 2 * self.agent_radius
        self.agent_bound_final = self.size - self.obstacle_threshold
        self.agent_bound = self.agent_bound_final
        self.angle_bound = np.pi

        # obstacle randomization bounds
        self.obstacle_size_high = params["obstacle_settings"]["size_high"]
        self.obstacle_bound = self.size - (2 * self.obstacle_threshold + self.obstacle_size_high)

        # goal randomization bounds
        self.goal_bound_init = 0
        self.goal_bound_final = self.size - 2 * self.agent_radius
        self.goal_bound = self.goal_bound_final

        # --- REWARD SCALE INITIALIZATION
        self.reward_scale_options       = reward_scale_options
        self.rew_dist_scale             = reward_scale_options.get("rew_dist_scale", 1)             if reward_scale_options else 1
        self.rew_dist_approach_scale    = reward_scale_options.get("rew_dist_approach_scale", 125)  if reward_scale_options else 125
        self.rew_head_scale             = reward_scale_options.get("rew_head_scale", 1)             if reward_scale_options else 1
        self.rew_head_approach_scale    = reward_scale_options.get("rew_head_approach_scale", 125)  if reward_scale_options else 125
        self.rew_obs_dist_scale         = reward_scale_options.get("rew_obs_dist_scale", 125)       if reward_scale_options else 125
        self.rew_goal_scale             = reward_scale_options.get("rew_goal_scale", 5000)          if reward_scale_options else 5000
        self.rew_obst_scale             = reward_scale_options.get("rew_obst_scale", -1000)         if reward_scale_options else -1000
        self.rew_time                   = reward_scale_options.get("rew_time", -0.25)               if reward_scale_options else -0.25

        # intialize the scaled reward components
        self.rew_head_scaled = 0
        self.rew_head_approach_scaled = 0
        self.rew_dist_scaled = 0
        self.rew_dist_approach_scaled = 0
        self.rew_obs_dist_scaled = 0

        # AGENT/TASK INITIALIZATION
        # --- define the uninitialized location of the agent and the target
        self._agent_loc = [-(self.size - 2*self.agent_radius), -(self.size - 2*self.agent_radius)]
        self._task_loc = [(self.size - 2*self.agent_radius), (self.size - 2*self.agent_radius)]
        
        # --- OBSTABLE INITIALIZATION
        self.obstacle_options = obstacle_options
        self.n_obstacles = obstacle_options.get("n_obstacles", 0)

        # define the buffers for spawning:
        self.obs_goal_buffer = self.goal_radius + 2 * self.obstacle_threshold + self.obstacle_size_high + self.allowance
        self.obs_agent_buffer = self.obstacle_threshold + self.obstacle_size_high + self.allowance
        self.obs_obs_buffer = 2 * self.obstacle_threshold + 2 * self.obstacle_size_high + 2 * self.allowance

        # get obstacle positions:
        self._obstacle_loc = self._set_obstacle_positions(goal_pos = self._task_loc, agent_pos = self._agent_loc)

        # --- whether an evaluation environment or not
        self.is_eval = is_eval
        
        # --- LOAD SIMULATION PARAMETERS AND INITIALIZE THE SIMULATION
        env =  MakeEnv(params)
        env.light_pos = np.array([0, 0, 1]) * 5 * self.size
        default_camera_config["distance"] = 3 * self.size
        env.make_env(agent_pos = self._agent_loc, 
                     task_pos = self._task_loc, 
                     n_rays = self.n_rays, 
                     obs_pos = self._obstacle_loc)
        self.model = env.model
        self.model.vis.global_.offwidth = width
        self.model.vis.global_.offheight = height
        self.data = mj.MjData(self.model)
        self.agent_id = self.model.body("agent").id
        self.goal_id = self.model.body("goal").id

        # --- ACTION SPACE INITIALIZATION
        self._set_action_space()

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        self.agent_init = np.zeros(3)
        self.agent_pose = np.append(self.data.xpos[self.agent_id][:2], self.data.qpos[2])

        # --- OBSERVATION SPACE INITIALIZATION
        self._set_observation_space()
        
        #--- pre allocate observation array once
        self._obs_buffer = np.zeros(self._obs_space_size, dtype = np.float32)
        self._lidar_buffer = np.zeros(self.n_ray_groups, dtype=np.float32)

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

    def _set_observation_space(self):
        ''' 
        internal method to set the bounds on the observation space

        order of the bounds:
            (2, ): the dx and dy components of how far the agent is from the goal
            (1, ): the distance between the agent and the goal
            (2, ): the cos(theta) and sin(theta) components of the agent's heading
            (2, ): the cos() and sin() componets of the relative bearing
            (2, ): local velocities of the agent (along local x and about local z)
            (n_ray_grous, ): minpooled groups of LiDAR scans

        '''
        # list of observation states
        obs_states = []
        obs_states += ["dx", "dy", "dgoal"]
        obs_states += ["c_theta", "s_theta", "c_psi", "s_psi"] 
        obs_states += ["v_lin", "v_ang"] 
        obs_states += [f"lidar_{n}" for n in range(self.n_ray_groups)]
        
        # define the obs_space_size:
        self._obs_space_size = len(obs_states)

        # index of each quantity
        for i, state in enumerate(obs_states):
            setattr(self,f"obs_{state}_idx",i)

        # initialize the scale on the observation space as being between [-1, 1]:
        low  = -np.ones((self._obs_space_size, ), dtype = np.float32) 
        high =  np.ones((self._obs_space_size, ), dtype = np.float32)

        # dx and dy bounds
        low[[self.obs_dx_idx, self.obs_dy_idx]]  = -2 * (self.size - self.agent_radius)
        high[[self.obs_dx_idx, self.obs_dy_idx]] =  2 * (self.size - self.agent_radius)

        # d_goal bounds 
        low[self.obs_dgoal_idx]    = 0
        high[self.obs_dgoal_idx]   = self.dmax

        # velocity bounds
        low[[self.obs_v_lin_idx, self.obs_v_ang_idx]]  = self.action_low * np.array([self.linear_scale, self.angular_scale], dtype = np.float32)
        high[[self.obs_v_lin_idx, self.obs_v_ang_idx]] = self.action_high * np.array([self.linear_scale, self.angular_scale], dtype = np.float32)
        
        # LiDAR bounds
        low[self.obs_lidar_0_idx:]  = 0.0
        high[self.obs_lidar_0_idx:] = self.dmax + np.sqrt(2 * self.agent_radius**2)

        # set the observation space:
        self.observation_space = gym.spaces.Box(
            low = low,            
            high = high,         
            dtype = np.float32)
        
        return self.observation_space
    
    def _set_action_space(self):
        ''' internal method to set the bounds on the agent's local x_linear and z_angular velocities'''
        # set the low and high of the action space:
        self.action_low = np.array([0.0, -1.0], dtype = np.float32)
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
            (1, ): the distance between the agent and the goal
            (2, ): the cos(theta) and sin(theta) components of the agent's heading
            (2, ): the cos() and sin() components of the relative bearing
            (n_ray_groups, ): minpooled groups of LiDAR scans
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
        rel_bearing = - ((bearing - heading + np.pi) % (2*np.pi) - np.pi)  
        c_bearing = np.cos(rel_bearing, dtype=np.float32)
        s_bearing = np.sin(rel_bearing, dtype=np.float32)

        # calculate the local velocities - linear and angular 
        # vx, vy, vz = self.data.qvel[0:3]
        # v_lin = np.sqrt(vx ** 2 + vy ** 2) / self.linear_scale
        # v_ang = vz / self.angular_scale
        v_lin = self.action_last[0]
        v_ang = self.action_last[1]
    
        lidar = self.data.sensordata[:-1]
        for i in range(self.n_ray_groups):
            self._lidar_buffer[i] = np.min(lidar[self._ray_per_group*i:self._ray_per_group*(i+1)])

        #--- modify obs buffer inplace instead of concatenation overhead (time + memory):
        self._obs_buffer[[self.obs_dx_idx, self.obs_dy_idx]] = dx, dy
        self._obs_buffer[self.obs_dgoal_idx] = np.sqrt(dx ** 2 + dy ** 2)
        self._obs_buffer[[self.obs_c_theta_idx, self.obs_s_theta_idx]] = c_theta, s_theta
        self._obs_buffer[[self.obs_c_psi_idx, self.obs_s_psi_idx]] = c_bearing, s_bearing
        self._obs_buffer[[self.obs_v_lin_idx, self.obs_v_ang_idx]] = v_lin, v_ang
        self._obs_buffer[self.obs_lidar_0_idx:]  = self._lidar_buffer

        self.agent_pose = np.append(self.data.xpos[self.agent_id][:2], self.data.qpos[2])

        return self._obs_buffer

    def _set_obstacle_positions(self,
                                agent_pos : list,
                                goal_pos  : list):
        """
        internal method for setting the position of the obstacles such that they:
            - are far enough from the borders of the arena
            - they are far enough from the position of the goal
            - they are far enough from the position of the agent
            - they are far enough from the position of other obstacles
        
        such that it can be ensured that no one obstacle spawns collided with another or that there is no navigable path between obstacles.

        :param agent_pos:   a list containing the coordinates of the agent in the worldbody in form ``[X, Y]``.
        :param goal_pos:    a list containing the coordinates of the goal in the worldbody in form ``[X, Y]``.

        :type agent_pos: list
        :type goal_pos: list
        
        """
        # need to first initialize a list of obstacle positions:
        obstacle_loc = []

        # initialize a list of things to check:
        spawn_queue = []

        # add the goal and the agent positions to the queue:
        spawn_queue.append(goal_pos)
        spawn_queue.append(agent_pos)

        # loop over the number of obstacles specified by the user:
        for i in range(self.n_obstacles):
            # define a bool for completion:
            satisfied = False

            # while the obstacle is not placed:
            while not satisfied:
                # generate an initial position for an obstacle: 
                candidate_obstacle = self.np_random.uniform(low = -self.obstacle_bound, high = self.obstacle_bound, size = 2)

                # counter for passes:
                passes = 0

                # check to see if the obstacle position is acceptable:
                for item, _ in enumerate(spawn_queue):
                    # determine which buffer to use:
                    if item == 0:
                        buffer = self.obs_goal_buffer
                    elif item == 1:
                        buffer = self.obs_agent_buffer
                    else:
                        buffer = self.obs_obs_buffer

                    # compute distance between candidate obstacle and item in spawn queue:
                    distance = np.linalg.norm(candidate_obstacle - spawn_queue[item])

                    # check to see if it passes:
                    if distance >= buffer:
                        passes += 1
                
                # check to see if we are satisfied with the placement:
                if passes == 2 + i:
                    satisfied = True

            # append this position to the queue of things to check, as well as the obstacle_loc list:
            spawn_queue.append(candidate_obstacle)
            obstacle_loc.append(candidate_obstacle)

        # return the obstacle locations:
        return obstacle_loc

    def reset_model(self, 
                    reset_randomize: bool = False):
        noise_low = -0.1
        noise_high = 0.1

        # get a copy of the initial_qpos
        # qpos = np.copy(self.init_qpos)
        # qvel = np.copy(self.init_qvel)

        qpos_array = np.copy(self.init_qpos)
        qvel_array = np.copy(self.init_qvel)

        if reset_randomize:
            qpos_array[0:2] = self.np_random.uniform(size = 2, low = -self.agent_bound, high = self.agent_bound) - self._agent_loc
            qpos_array[2]   = self.np_random.uniform(size = 1, low = -self.angle_bound, high = self.angle_bound) % (2 * np.pi)
            qpos_array[3:5] = self.np_random.uniform(size = 2, low = -self.goal_bound, high = self.goal_bound) - self._task_loc

            # need to get the new position of the agent in the worldbody:
            agent_pos = qpos_array[0:2] + self._agent_loc

            # need to get the new position of the goal in the worldbody:
            goal_pos = qpos_array[3:5] + self._task_loc

            # generate new positions for the obstacles:
            obstacle_locs = self._set_obstacle_positions(agent_pos = agent_pos, goal_pos = goal_pos)

            # set the new position of these obstacles:
            for i in range(self.n_obstacles):
                qpos_array[5+2*i:7+2*i] = obstacle_locs[i] - self._obstacle_loc[i]

            qvel_array[0:2] = self.np_random.uniform(size = 2, low = noise_low, high = noise_high)

            self.init_qpos = qpos_array
            self.init_qvel = qvel_array
                
        # if self.is_eval and self.render_mode=="human":
        #     qpos[:2] = self.agent_pose[:2] - self._agent_loc if (not self.collision) else self.init_qpos[:2]
        #     qpos[2] = self.agent_pose[2] if (not self.collision) else self.init_qpos[2]
        #     qvel[:3] = np.zeros(3)

        # TODO - implemented to penalize large changes in control actions. Remove if not effective
        self.action_last = qvel_array[0:2]

        self.set_state(qpos_array, qvel_array)
        ob = self._get_obs()

        # get the previous d_goal:
        dx, dy = ob[[self.obs_dx_idx, self.obs_dy_idx]]
        self.d_goal_last = ob[self.obs_dgoal_idx]       # to track distance approach progress
        self.d_init = self.d_goal_last                  # to track overall distance progress
   
        # get the initial abs_diff
        c_bearing, s_bearing = ob[[self.obs_c_psi_idx, self.obs_s_psi_idx]]
        self.prev_abs_diff = abs(np.arctan2(s_bearing, c_bearing, dtype=np.float32))
        self.abs_diff_init = self.prev_abs_diff

        # get the initial min_dist:
        lidar_obs = ob[self.obs_lidar_0_idx:]
        min_dist = min(lidar_obs)
        self.min_dist_last = min_dist

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
        

        # --- CHECK RANDOMIZATION CONDITIONS
        if not self.is_eval:
            if self.episode_counter % self.randomization_freq == 0:
                self.reset_randomize = True

            # reset mujoco model:
            ob = self.reset_model(reset_randomize=self.reset_randomize)
        
        else: # if an evaluation env, always reset and with the largest bound possible
            self.agent_bound = self.agent_bound_final
            self.goal_bound = self.goal_bound_final
            ob = self.reset_model(reset_randomize=True)

        # reset flags:
        self.reset_randomize = False

        # render if mode == "human":
        if self.render_mode == "human":
            self.render()
        return ob, {}

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
        nobs = self._get_obs()
        dx, dy                  = nobs[[self.obs_dx_idx, self.obs_dy_idx]]
        d_goal                  = nobs[self.obs_dgoal_idx]
        c_theta, s_theta        = nobs[[self.obs_c_theta_idx, self.obs_s_theta_idx]]
        c_bearing, s_bearing    = nobs[[self.obs_c_psi_idx, self.obs_s_psi_idx]]
        v_lin, v_ang            = nobs[[self.obs_v_lin_idx, self.obs_v_ang_idx]]
        lidar_obs               = nobs[self.obs_lidar_0_idx:]

        # get the minimum LiDAR reading:
        min_dist = min(lidar_obs)

        # get the difference between the agents current heading, and the required heading:
        heading  = np.arctan2(s_theta, c_theta) % (2 * np.pi)
        bearing  = np.arctan2(s_bearing, c_bearing)
        abs_diff = abs(bearing)

        # 3. termination conditions: 
        # when the agent is close to the goal:
        distance_cond = d_goal < self.distance_threshold

        # when the agent is close to obstacles:
        obstacle_cond = np.min(lidar_obs) < self.obstacle_threshold
        
        # when the agent has not reduced the d_goal for N steps, where N is 200:
        # if d_goal > self.d_goal_last: 
        #     self.dist_progress_count += 1
        # else:
        #     self.dist_progress_count = 0

        # # when the abs_diff is more than 15 degrees and still growing:
        # if abs_diff >= self.prev_abs_diff and (abs_diff > np.deg2rad(15)):
        #     self.head_progress_count += 1
        # else:
        #     self.head_progress_count = 0
        
        term = distance_cond or obstacle_cond 
        # or (self.dist_progress_count >= self.progress_threshold) or (self.head_progress_count >= self.progress_threshold)
        
        info = {}

        if term:
            info["is_success"]          = bool(distance_cond)        # if successful
            info["obstacle_cond"]       = bool(obstacle_cond)        # if not successful

            # these are the progress conditions:
            info["dist_progress_cond"]  = (self.dist_progress_count >= self.progress_threshold)
            info["head_progress_cond"]  = (self.head_progress_count >= self.progress_threshold)

            # flag for collision:
            self.collision = bool(obstacle_cond)
            
        # 4. reward:
        if distance_cond:
            rew = self.rew_goal_scale
        elif obstacle_cond:
            rew = self.rew_obst_scale
        else:
            #--- DISTANCE REWARD:
            # this reward term incentivizes closing the distance between the agent and the goal:
            # rew_dist = (self.d_init - d_goal) / self.d_init         # distance reward relative to the starting state 
            # self.rew_dist_scaled = self.rew_dist_scale * rew_dist

            #--- DISTANCE APPROACH REWARD:
            # this reward term incentivizes approaching the goal, and rewards 0 otherwise:
            rew_dist_approach = max((self.d_goal_last - d_goal), 0)
            self.rew_dist_approach_scaled = rew_dist_approach * self.rew_dist_approach_scale

            #---  BASE HEADING REWARD:
            # penalize based on the absolute difference in heading:
            # if action[0] >= 0.05:                 # gated by forward velocity
            # if (self.d_goal_last - d_goal) > 1e-5:  # gated by making progress toward the goal 
            #     rew_head = 1.0 - np.tanh(abs_diff / np.pi)

            #     # bonus reward for being within +/- 5 degree of the desired trajectory:
            #     # angle_threshold_rad = np.deg2rad(2.5)
            #     # if abs_diff <= angle_threshold_rad:
            #     #     rew_head += 1 - 1 / angle_threshold_rad * abs_diff
            # else: rew_head = 0

            # scale reward:
            # self.rew_head_scaled = self.rew_head_scale * rew_head

            #--- HEADING APPROACH REWARD:
            # this reward term incentivizes approaching the required heading, and rewards 0 otherwise
            rew_head_approach = max((self.prev_abs_diff - abs_diff), 0)
            self.rew_head_approach_scaled = self.rew_head_approach_scale * rew_head_approach

            #--- CHANGE IN MINIMUM OBSTACLE DISTANCE REWARD TERM:
            rew_obs_dist_change = max((self.min_dist_last - min_dist), 0)
            self.rew_obs_dist_change_scaled = self.rew_obs_dist_scale * rew_obs_dist_change
            # min_dist = min(lidar_obs)
            # rew_obs_dist = min_dist / self.dmax
            # self.rew_obs_dist_scaled = 

            #--- TOTAL REWARD TERM:
            rew = self.rew_head_scaled + self.rew_head_approach_scaled + self.rew_dist_scaled + self.rew_dist_approach_scaled + self.rew_time + self.rew_obs_dist_change_scaled

            # print to user:
            if self.render_mode == "human":
                # observation debug
                print(f" @ episode {self.episode_counter} | "
                      f"(dx,dy)=({dx: 4.2f},{dy: 4.2f}) | d_goal={d_goal:5.3f}  | "
                      fr"θ={heading/np.pi*180: 6.2f} | Ψ={bearing/np.pi*180: 6.2f} | "
                      f"(vx, vz)_(t-1)=({v_lin: 5.3f},{v_ang: 5.3f}) | "
                      f"(vx, vz)_t={action[0]: 5.3f},{action[1]: 5.3f}          ", end="\r")

                # # lidar debug
                # print(lidar_obs)

                # reward debug
                print(f" @ episode {self.episode_counter} | "
                      f"rew_dist: {self.rew_dist_scaled: 6.4f} | "
                      f"rew_dist_app: {self.rew_dist_approach_scaled: 6.4f} | "
                      f"rew_head: {self.rew_head_scaled: 6.4f} | "
                      f"rew_obs_dist: {self.rew_obs_dist_change_scaled: 6.4f} |"
                      f"rew_head_app: {self.rew_head_approach_scaled: 6.4f}        ",
                      end="\r")
                pass
                
        # advance histories:
        self.d_goal_last = d_goal
        self.prev_abs_diff = abs_diff
        self.min_dist_last = min_dist
        self.action_last = action

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