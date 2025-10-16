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
        size = params["ground_settings"]["internal_length"]
        agent_radius = params["agent_settings"]["radius"]
        
        # --- define the uninitialized location of the agent and the target
            # TODO - to be set randomly in the reset() method
        self._agent_loc = np.array([0, 0], dtype=np.float32)
        self._task_loc = np.array([0.5, 0.5], dtype=np.float32)

        # --- define the observation space
        self.observation_space = gym.spaces.Box(
            low=-size/2,        # [x_min, y_min, target_x_min, target_y_min]
            high=size/2,        # [x_max, y_max, target_x_max, target_y_max]
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
        self.distance_threshold = 0.01
        self.obstacle_threshold = 0.05 + agent_radius

    # TODO - create a _get_info() methods
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
        obstacle_cond = min(lidar_obs) < self.obstacle_threshold
        term = distance_cond or obstacle_cond
        
        # 4. reward - TODO - placeholder for reward values
        if distance_cond:
            rew = 200
        elif obstacle_cond:
            rew = -100
        else:
            rew = d_goal

        # 5. info (optional)

        # 6. render if render_mode human 
        # TODO - test the gymnasium mujoco render
        if self.render_mode == "human":
            self.render()

        return nobs, rew, term
    
    # TODO - create the close() method