from typing import Sequence
import numpy as np 
import gymnasium as gym
import mujoco
from gymnasium.envs.mujoco import MujocoEnv

class TestingEnv(MujocoEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    # TODO - actually create the render somehow

    def __init__(self, 
                 act_bound: Sequence[int|float], # either a list or tuple of float of x_linear bound, y_linear bound (optional) and z_angular bound
                 size: float = 5.0,
                 render_mode: str | None = None,
                 width: int = 480,
                 height: int = 480,
                 camera_id: int | None = None,
                 camera_name: str | None = None,
                 default_camera_config: dict[str, float | int] | None = None
                ):
        
        """ Observation and Action Spaces """
        self.size = size       # assume 5x5 m^2 environment
        
        # --- ensure all elements of act_bound are numeric
        if not all(isinstance(x, (int, float)) for x in act_bound):
            raise TypeError("All elements in act_bound must be int or float")

        # --- unpack action_bounds and check length of argument
        if len(act_bound) == 2:
            self._is_holonomic = False
        elif len(act_bound) == 3:
            self._is_holonomic = True
        else:
            raise ValueError("Action bounds must have eiher 2 or 3 elements")


        # --- unpack the action bounds
        if not self._is_holonomic:
            self.act_x_linear_bound, self.act_z_angular_bound = map(abs, map(float, act_bound))
        else:
            self.act_x_linear_bound, self.act_y_linear_bound, self.act_z_angular_bound = map(abs, map(float,act_bound))
            
        
        # --- define the uninitialized location of the agent and the target - to be set randomly in the reset() method
        self._agent_loc = np.array([0, 0], dtype=np.float32)
        self._target_loc = np.array([0, 0], dtype=np.float32)

        # --- define the observation space
        self.observation_space = gym.spaces.Box(
            low=-size/2,        # [x_min, y_min, target_x_min, target_y_min]
            high=size/2,        # [x_max, y_max, target_x_max, target_y_max]
            shape=(4,), dtype=np.float32)
        
        # --- define the action space (continuous linear and angular velocity)
        if not self._is_holonomic:
            self.action_space = gym.spaces.Box(
                low = np.array([-self.act_x_linear_bound, -self.act_z_angular_bound], dtype=np.float32),
                high = np.array([self.act_x_linear_bound, self.act_z_angular_bound], dtype=np.float32),
                dtype = np.float32
            )
        else:
            self.action_space = gym.spaces.Box(
                low = np.array([-self.act_x_linear_bound, -self.act_y_linear_bound, -self.act_z_angular_bound]),
                high = np.array([self.act_x_linear_bound, self.act_y_linear_bound, self.act_z_angular_bound]),
                dtype = np.float32
            )

        """ Render mode stuff """
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

        # TODO - load the MuJoCo MJCF/URDF model and create a simulation object MjData
        self.model = mujoco.MjModel.from_xml_path("assets/env_test_minh.xml")
        self.data = mujoco.MjData(self.model)

        # TODO - initialize the renderer
        from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
        self.renderer = MujocoRenderer(
            self.model,
            self.data,
            default_cam_config=default_camera_config,
            width=width,
            heigh=height,
            camera_id=camera_id,
            camera_name=camera_name
        )

    # TODO - create a _get_obs() and _get_info() methods
    def _get_obs():
        ''' Function to obtain the LiDAR simulation scan and location of agent/goal at any instance '''
        pass
    # TODO - create the reset() method
    # TODO - create the step() method
    def step(self, action):
        # 1. move the simulation forward with the TRANSFORMED action (w.r.t. original frame)
        # 2. collect the new observation (LiDAR simulation, location of agent/goal using the custom _get_obs())
        # 3. termination condition
        # 4. reward
        # 5. info (optional)
        # 6. render if render_mode human
        pass
    # TODO - create the render() method
    # TODO - create the close() method