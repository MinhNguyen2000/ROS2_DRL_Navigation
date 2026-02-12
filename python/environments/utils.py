from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from collections import deque

# Packages for parsing tensorboard log
from dataclasses import dataclass
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Sequence, Dict
from tbparse import SummaryReader

default_reward_scale = {
    "rew_dist_scale" : 0.0,               
    "rew_dist_approach_scale" : 200.0,
    "rew_head_scale" : 0.0,
    "rew_head_approach_scale" : 200.0,  
    "rew_goal_scale" : 3_000.0,          
    "rew_obst_scale" : -250.0,
    "rew_obs_dist_scale" : 0.0,
    "rew_obs_align_scale" : 0.5,        
    "rew_time" : -0.5} 

default_randomization_options = {"randomization_freq": 1}

default_obstacle_options = {"n_obstacles": 0}

def make_env(n_proc: int = 12,
            seed: int = 42,
            max_episode_steps: int = 3_000,
            reward_scale: dict = default_reward_scale,
            randomization_options: dict = default_randomization_options,
            obstacle_options: dict = default_obstacle_options,
            normalize: bool=True,
            n_ray_groups: int = 18):

        print("Making subprocess vectorized environments!")
        env = make_vec_env("Nav2D-v0", 
                            n_envs=n_proc, 
                            seed=seed,
                            env_kwargs={"max_episode_steps": max_episode_steps,
                                        "reward_scale_options": reward_scale,
                                        "randomization_options": randomization_options,
                                        "obstacle_options": obstacle_options,
                                        "n_ray_groups" : n_ray_groups
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
        n_ray_groups            = old_env.get_attr("n_ray_groups")[0]

        if old_env is not None:
            old_env.close()

        # CREATE AND SET NEW ENVIRONMENT
        n_obstacles = self.obstacle_per_level * self.env_level
        obstacle_options = {"n_obstacles": n_obstacles}

        env = make_env(n_proc=num_envs,
                        reward_scale=reward_scale_option,
                        randomization_options=randomization_options,
                        obstacle_options=obstacle_options,
                        n_ray_groups = n_ray_groups)
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

@dataclass
class RunData:
    run_name: str
    df: pd.DataFrame
    tag: str
    
class TensorBoardParser():
    def __init__(self, 
                 log_dirs: Sequence[str | Path],
                 tags: str="rollout/success_rate"):
        '''
        Docstring for __init__
        
        :param log_dirs: Path(s) to the tensorboard log file(s)
        :param tags: Metrics to extract from the tensorboard log

        :type log_dirs: Sequence[str | Path]
        :type tags: str
        '''
        self.log_dirs = [Path(p) for p in log_dirs]
        self.tag = tags
        self.runs: List[RunData] = []

    def load(self):
        '''
        Parse the training metric data from the TensorBoard logs of the specified runs 
        and store them as a list of RunData objects
        '''

        runs: List[RunData] = []
        for log_dir in self.log_dirs:
            reader =SummaryReader(log_dir)
            df = reader.scalars

            run_name = log_dir.name
            
            # filter by the corresponding tag (e.g. rows with tag==rollout/success_rate)
            tag_df = df[df['tag'] == self.tag].copy()

            # handle duplicated steps, only keeping the first value in this case
            unique_tag_df = tag_df.loc[~tag_df['step'].duplicated(keep="first")]

            runs.append(
                RunData(
                    run_name=run_name,
                    df=unique_tag_df[["step", "value"]].reset_index(drop=True),
                    tag=self.tag
                )
            )

        self.runs = runs

    def build_grid(self, step_interval: int = 5_000):
        '''
        Build a common regularly spaced grid as the x axis for the final plot
        
        :param step_interval: The interval between steps
        :type step_interval: int
        '''
        if self.runs is None:
            raise ValueError("No run loaded; call .load() first")
        
        # find the minimum and maximum step value across the runs
        min_step = min([min(run.df['step']) for run in self.runs])
        max_step = max([max(run.df['step']) for run in self.runs])
    
        # common grid
        return np.arange(start=min_step, stop = max_step+step_interval, step=step_interval)

    def interp_on_grid(self,
                       grid: np.ndarray,
                       method: str = 'linear'):
        '''
        Inter/Extrapolate the values from the existing time steps in each run to those of the common grid
        
        :param self: Description
        :param method: Method of interpolation ('linear', 'index', ...)
        '''
        if self.runs is None:
            raise ValueError("No run loaded; call .load() first")
        

        aligned_cols: Dict[str, pd.Series] = {}
        
        # process the values at each run in the aligned_cols dict
        for run in self.runs:
            df=run.df.set_index('step').sort_index()      # use the step number as the df index

            # create a sorted full index list including the original df indices and the common regular indices
            full_index = np.unique(np.concatenate([df.index, grid]))

            # reindex the original df with the big index list, filling missing values (at the time steps of the common grid) with NaN
            reindexed_df = df.reindex(full_index)

            # interpolate the missing values using the neighboring values
            # border values are populated by forward/backward fills
            interpolated_df = reindexed_df['value'].interpolate(method=method).ffill().bfill()

            # draw the values at the common grid indices to equalize the values of each run
            aligned_cols[run.run_name] = interpolated_df.loc[grid]

        # turn the aligned_cols dict created above to a final df
        aligned_df = pd.DataFrame(aligned_cols, index=grid)
        return aligned_df

    def aggregate(
            self, 
            aligned: pd.DataFrame,
            band: str = "minmax"
        ):
        '''
        Compute the mean and bands (min/max or mean±std) accross runs and return a DataFrame with columns step, mean, low, high
        
        :param self: Description
        :param aligned: Dataframe with data from each runs with the indices aligned to a common grid
        :param band: Method of displaying the shaded min-max region (`minmax` for raw min-max values and `std` for mean±std lower and upper bounds)

        :type aligned: pd.DataFrame
        :type band: str
        '''

        mean = aligned.mean(axis=1)     # axis=1 find statistics across runs (columns)

        if band == "minmax":
            low = aligned.min(axis=1)
            high = aligned.max(axis=1)
        elif band == "std":
            std = aligned.std(axis=1)
            low = mean-std
            high = mean+std
        else:
            raise ValueError("band must be 'minmax' or 'std'")
        
        return pd.DataFrame(
            {
                'step': aligned.index.values,
                'mean': mean.values,
                'low': low.values,
                'high': high.values
            }
        )

    
    def run(self,
            step_interval: int = 5_000,
            method: str = 'linear',
            band: str = 'minmax'):
        
        '''
        Compute the mean and min/max statistics at the common steps across different runs
        
        :param self: Description
        :param method: The linear interpolation method, default to linear interpolation (read pandas df.interpolate for more method)
        :param band: Method for reporting the min-max band ('minmax' for raw values, 'std' for mean±std lower/upper bounds)

        :type method: str
        :type band: str
        '''
        self.load()
        common_grid = self.build_grid(step_interval=step_interval)
        aligned = self.interp_on_grid(grid=common_grid, method=method)

        return self.aggregate(aligned=aligned, band=band)