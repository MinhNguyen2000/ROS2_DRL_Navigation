# import the MakeEnv class written by Matt
# from environments import MakeEnv

from typing import Sequence
import numpy as np 
import gymnasium as gym
import mujoco as mj
from gymnasium.envs.mujoco import MujocoEnv
import json
from model_creation import MakeEnv

class Nav2D(MujocoEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 500}

    def __init__(self, 
                 json_file: str = "environment_params.json",
                 frame_skip: int = 2,
                 default_camera_config: dict[str, float | int] | None = None,
                 render_mode: str | None = None,
                 width: int = 480,
                 height: int = 480,
                 camera_id: int | None = None,
                 camera_name: str | None = None,
                ):
        ''' class constructor to initialize the environment (Mujoco model and data), the observation space, and renderer
        
        Arguments:
            json_file:              a string that containes the name of the environment parameters json file, which
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
        # self.size = size       # assume 5x5 m^2 environment

        self.width, self.height = width, height
        self.render_mode = render_mode
        self.camera_name = camera_name
        self.camera_id = camera_id
        self.frame_skip = frame_skip

        # --- load the simulation parameters
        # TODO - handle full path expansion of json file
        with open(json_file) as f:
            params = json.load(f)
        self.size = params["ground_settings"]["internal_length"]
        self.agent_radius = params["agent_settings"]["radius"]
        
        # --- define the uninitialized location of the agent and the target
        self._agent_loc = np.array([0, 0], dtype=np.float32)
        self._task_loc = np.array([-0.5, -0.5], dtype=np.float32)

        # --- define the observation space
        self.observation_space = gym.spaces.Box(
            low=-self.size/2,        # [x_min, y_min, target_x_min, target_y_min]
            high=self.size/2,        # [x_max, y_max, target_x_max, target_y_max]
            shape=(4,), dtype=np.float32)
        
        # --- load simulation params and initialize the simulation
        env =  MakeEnv(params)
        env.make_env(agent_pos = self._agent_loc, 
                     task_pos = self._task_loc, 
                     n_rays = 36)
        self.model = env.model
        self.model.vis.global_.offwidth = width
        self.model.vis.global_.offheight = height
        self.data = mj.MjData(self.model)

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

        # --- initialize the renderer
        if "render_fps" in self.metadata:
            assert (
                int(np.round(1.0 / self.dt)) == self.metadata["render_fps"]
            ), f'Expected value: {int(np.round(1.0 / self.dt))}, Actual value: {self.metadata["render_fps"]}'

        from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

        self.mujoco_renderer = MujocoRenderer(
            self.model,
            self.data,
            default_cam_config = default_camera_config,
            width           = self.width,
            height          = self.height,
            max_geom        = 1000,
            camera_id       = self.camera_id,
            camera_name     = self.camera_name,
            visual_options  = {},
        )

        self.window = None
        self.clock = None

        self.rot_matrix = np.zeros((2,2))
        self.agent_id = self.model.body("agent").id
        self.goal_id = self.model.body("goal").id

        # --- termination conditions
        self.distance_threshold = 0.05
        self.obstacle_threshold = 0.05 + self.agent_radius

    # TODO - create a _get_info() methods
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
    
    def reset_model(self):
        noise_low = -0.1
        noise_high = 0.1

        agent_bound = self.size - 2*self.agent_radius
        angle_bound = np.pi
        goal_bound = self.size - self.agent_radius

        qpos = np.copy(self.init_qpos)
        qpos[0:2] += self.np_random.uniform(size=2, low=-agent_bound, high=agent_bound)
        qpos[2] += self.np_random.uniform(size=1, low=-angle_bound, high=angle_bound)
        qpos[3:5] += (- self._task_loc + self.np_random.uniform(size=2, low=-goal_bound, high=goal_bound))


        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=noise_low, high=noise_high
        )

        self.set_state(qpos, qvel)
        pass

    def reset(self,
              seed: int | None = None,
              options: dict | None = None):
        # TODO - add flags for when to randomize goal/agent/obstacles
        mj.mj_resetData(self.model, self.data)

        ob = self.reset_model()
        info = None

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
        return np.sqrt(np.square(point_a[0]-point_b[0]) + np.square(point_a[1]-point_b[1]))

    # TODO - create the step() method
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
        obstacle_cond = min(lidar_obs) < self.obstacle_threshold

        term = distance_cond or obstacle_cond
        
        # 4. reward
        if distance_cond:
            rew = 200.0
        elif obstacle_cond:
            rew = -100.0
        else:
            rew = -d_goal

        # 5. info (optional)

        # 6. render if render_mode human 
        # TODO - test the gymnasium mujoco render
        if self.render_mode == "human":
            self.render()

        return nobs, rew, term
    
    # TODO - create the close() method