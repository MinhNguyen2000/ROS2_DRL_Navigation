from typing import Sequence
import numpy as np 
import gymnasium as gym
import mujoco
from gymnasium.envs.mujoco import MujocoEnv

class TestingEnv(MujocoEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    # TODO - actually create the render somehow

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
            xml_file=xml_file,
            frame_skip=frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config,
            **kwargs
        )

        self.window = None
        self.clock = None

    # TODO - create a _get_obs() and _get_info() methods
    def _get_obs():
        ''' Function to obtain the LiDAR simulation scan and location of agent/goal at any instance '''
        pass
    # TODO - create the reset() method
    # TODO - create the step() method
    def step(self, action):
        # 1. move the simulation forward with the TRANSFORMED action (w.r.t. original frame)
        # assuming the action is a (2,1) specifying (x_dot, theta_dot)
        rot_matrix = [[np.cos(), -np.sin()],
                      [np.sin(), np.cos()]]
        self.do_simulation(action, self.frame_skip)

        # 2. collect the new observation (LiDAR simulation, location of agent/goal using the custom _get_obs())\
        observation = self._get_obs()
        # 3. termination condition
        # 4. reward
        # 5. info (optional)
        # 6. render if render_mode human
        pass
    # TODO - create the render() method
    # TODO - create the close() method