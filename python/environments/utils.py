from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from collections import deque

default_reward_scale = {
        "rew_dist_scale":           0.0,
        "rew_dist_approach_scale":  75.0,
        "rew_head_scale":           0.0,
        "rew_head_approach_scale":  75.0,
        "rew_time":                 -0.5,
        "rew_goal_scale":           3_000.0,
        "rew_obst_scale":           -1_000.0}

default_randomization_options = {"randomization_freq": 1}

default_obstacle_options = {"n_obstacles": 0}

def make_env(n_proc: int = 12,
            seed: int = 73,
            max_episode_steps: int = 2_000,
            reward_scale: dict = default_reward_scale,
            randomization_options: dict = default_randomization_options,
            obstacle_options: dict = default_obstacle_options,
            normalize: bool=True):

        print("Making subprocess vectorized environments!")
        env = make_vec_env("Nav2D-v0", 
                            n_envs=n_proc, 
                            seed=seed,
                            env_kwargs={"max_episode_steps": max_episode_steps,
                                        "reward_scale_options": reward_scale,
                                        "randomization_options": randomization_options,
                                        "obstacle_options": obstacle_options
                                        },
                            vec_env_cls=SubprocVecEnv, 
                            vec_env_kwargs=dict(start_method='forkserver'))

        if normalize: env = VecNormalize(env, training=True, norm_obs=True, norm_reward=True)

        return env

class AdvanceEnvCallback(BaseCallback):
    """
    This callback advances training to more complicated environment with increasing number of obstacles
    """
    def __init__(self, success_window: int = 50, success_threshold: float = 0.9, obstacle_per_level: int = 10):
        ''' Intialize the callback object

        :param success_window:      size of the window of previous training episodes to calculate the average success rate
        :param success_threshold:   minimum success rate to advance the curriculum

        :type success_window:       int
        :type success_threshold:    float
        '''
        super().__init__()
        self.success_window = success_window
        self.success_threshold = success_threshold
        self.success_buffer = deque(maxlen=success_window)

        self.env_level = 0
        self.max_level = 4
        self.obstacle_per_level = obstacle_per_level
        self.threshold_drop_per_level = 0.05
        self.n_obstacles = self.obstacle_per_level * self.env_level

        self.term_cond = {
            "is_success":           0,
            "obstacle_cond":        0,
            "TimeLimit.truncated":  0
        }

        self.env_advance = False            # flag to decide whether to advance the environment at the end of a model.learn()

    def _on_step(self):
        """
        Triggered every training step and do the following
        1.  Access the dones and infos of the vectorized environments to check terminal episodes. If there 
            is a terminated environment then track success and failure modes
        2.  Once the success window is filled, determine the success rate.
        3.  If the success rate exceeds the success threshold and the env level isn't the max level, increase the env level and 
            set the ```env_advance``` flag to True to trigger the env advancement at the end of the current model.learn()
        """
        # OBTAIN INFO FROM ALL ENVIRONMENTS
        dones = self.locals["dones"]
        infos = self.locals["infos"]

        # CHECK TERMINAL EPISODES AND UPDATE SUCCESS/FAILURE COUNTS
        for i, (done, info) in enumerate(zip(dones, infos)):
            if done:
                self.success_buffer.append(info.get("is_success", False))
                for k in self.term_cond.keys(): self.term_cond[k] += int(info.get(k, False))
        
        # ADVANCE ENVIRONMENT BASED ON SUCCESS RATE
        if len(self.success_buffer) >= self.success_window:
            success_rate = sum(self.success_buffer) / len(self.success_buffer)

            print(f"Current train success: {success_rate*100:5.2f}%", end="\r")
            
            # increase environment level if: 1) the success rate passes, and 2) the env level is not already maxed out
            if success_rate >= self.success_threshold and self.env_level < self.max_level:
                self._report_termination_cond()
                self.env_level = min(self.env_level + 1, self.max_level)
                print(f"Increased the environment level to {self.env_level}")
                self.env_advance = True
                self.success_buffer.clear()
        return True
    
    def _on_training_end(self):
        """Triggered before exiting the learn() method to do the following
        1.  Get the current training env and extract the current reward_scale_options and randomization_options for the new env
        2.  Close the old env (important since this will release the subprocesses in previous training)
        3.  Increase the number of obstacles according to the current env level and update the obstacle_options dict
        4.  Make the new training environment with the reward_scale_options, randomization_options, and obstacle_options and set
            this as the new training environment for the agent.
        5.  Change the env_advance flag back to False
        """
        if not self.env_advance:
            return
        
        # DESTROY OLD ENVIRONMENT (to release the subprocesses)
        old_env     = self.model.get_env()
        num_envs    = old_env.num_envs
        reward_scale_option     = old_env.get_attr("reward_scale_options")[0]
        randomization_options   = old_env.get_attr("randomization_options")[0]
        if old_env is not None:
            old_env.close()

        # CREATE AND SET NEW ENVIRONMENT
        n_obstacles = self.obstacle_per_level * self.env_level
        obstacle_options = {"n_obstacles": n_obstacles}

        env = make_env(n_proc=num_envs,
                        reward_scale=reward_scale_option,
                        randomization_options=randomization_options,
                        obstacle_options=obstacle_options)
        env.reset()
        self.model.set_env(env)
        model_env = self.model.get_env()

        # # reduce the success threshold per level
        # self.success_threshold -= self.threshold_drop_per_level

        # debug print to verify the number of obstacles in the new environment
        print(f"Environment level {self.env_level:3d} | "
                f"Number of obstacles {model_env.get_attr('n_obstacles')[0]:3d} | "
                f"Success threshold {self.success_threshold: 5.3f}")
        
        # Reset the environment advance flag
        self.env_advance = False
    
    def _report_termination_cond(self):
        success_count  = self.term_cond.get('is_success',            'no is_success')
        obstacle_count = self.term_cond.get('obstacle_cond',         'no obstacle_cond')
        trunc_count    = self.term_cond.get('TimeLimit.truncated',   'no TimeLimit.truncated')
        eps_sum = success_count + obstacle_count + trunc_count

        print(f"\nTraining result reported from environment level {self.env_level} with "
                f"success={success_count} ({success_count/eps_sum*100:5.2f}%) | "
                f"obstacle={obstacle_count} ({obstacle_count/eps_sum*100:5.2f}%) | "
                f"trunc={trunc_count} ({trunc_count/eps_sum*100:5.2f}%)")