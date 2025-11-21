# ===================================================================
# This script is used for training TD3 using stable baselines 3 with 
# SubprocVecEnv for multi-core processing, with either the "forkserver" 
# or the "spawn" method

# These methods only work when the training code is wrapped in a 
# if __name__ == "main" block
# ===================================================================

from stable_baselines3 import TD3,SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise

import torch
import gymnasium as gym
from gymnasium.wrappers import RescaleAction
import nav2d        # Have to import the nav2d Python script, else we can't make env
import nav2d_testing
import numpy as np
import os, re, json, time
from datetime import datetime
from tqdm import tqdm
from collections import deque

# Ignore User Warnings (for creating a new folder to save policies)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import optuna


def main():
    def eval_policy(env: gym.Env, 
         num_evals: int, 
         model):
        # reward list:
        eval_rew_hist = []

        # for each episode in the num_evals:
        for _ in range(num_evals):
            obs, _ = env.reset()
            done = False

            # initialize episodic reward:
            eval_rew = 0

            # while False:
            while not done:
                # get action and step:
                with torch.no_grad():
                    action, _ = model.predict(obs, deterministic = True)
                    nobs, reward, term, trunc, _ = env.step(action)
                    done = term or trunc
                    
                    # advance reward:
                    eval_rew += reward

                    # advance observation, reset if not:
                    obs = nobs if not done else env.reset()
        
            # append:
            eval_rew_hist.append(eval_rew)

        return np.mean(eval_rew_hist).round(3)

    class CurriculumCallback(BaseCallback):
        def __init__(self, success_window = 100, threshold = 0.9):
            super().__init__()
            self.success_threshold = threshold
            self.success_window = success_window
            self.buffer = deque(maxlen=success_window)
            self.goal_bound_levels = np.linspace(0,1,9)
            self.goal_level_idx = 0

            self.explore_boost_steps = 0            # keep count of 

        def _on_step(self) -> bool:
            # grab info dict from each parallel env
            infos = self.locals["infos"]

            for info in infos:
                if "is_success" in info:        # check if terminal state in each env
                    self.buffer.append(info["is_success"])

            if len(self.buffer) >= self.success_window:
                success_rate = sum(self.buffer)/len(self.buffer)
                print(f"Current training success {success_rate:3.2f}", end="\r")

                # if the success rate of the past N episodes exceed success threshold, progress with curriculum for all env
                if success_rate > self.success_threshold:
                    self._advance_curriculum()
                    self.buffer.clear()

            # handle episodes with increased action noise
            if self.explore_boost_steps > 0:
                self.explore_boost_steps -= 1
                if self.explore_boost_steps == 0:
                    
                    vec_noise = self.model.action_noise
                    for i in vec_noise.noises:
                        i._sigma = self.noise_std_backup.copy()
                    
                    print(f"\nFinished with steps of increased exploration, noise reset to {i._sigma}")
            return True     # to continue training, return True, else return False
    
        def _advance_curriculum(self):
            ''' Function to advance the curriculum by
            1. increasing the goal randomization bound 
            2. boost action_noise to increase exploration for a set number of steps 
            3. increasing the success window and buffer size to ensure performance consistency
            '''

            #--- increase the goal bound
            env = self.model.get_env()  # vecnormalized wrapped vec env
            vec = env.venv              # unwrap to vec env
            self.goal_level_idx += 1
            print(f"\nIncreasing goal bound - level {self.goal_level_idx:2d} | {self.goal_bound_levels[self.goal_level_idx]:5.3f}")
            self.goal_level_idx = min(self.goal_level_idx, len(self.goal_bound_levels) - 1)

            vec.env_method("_set_goal_bound", self.goal_bound_levels[self.goal_level_idx])

            #--- boost exploration for several steps
            self.explore_boost_steps = 2_000
            vec_noise = self.model.action_noise             # VectorizedActionNoise, which is a vector of the action noise (NormalActionNoise)
            noise = vec_noise.noises                        # list of NormalActionNoise instance of one environment
            self.noise_std_backup = noise[0]._sigma.copy()  # keep a copy of the original noise stdev

            for i in noise:
                i._sigma *= 2.0     # double the action noise stdev for all child env
            print(f"Action noise will be increased to {i._sigma} for {self.explore_boost_steps} steps")

            #--- increase the success window and success buffer length
            self.success_window = int(self.success_window*1.1)
            self.buffer = deque(maxlen=self.success_window)
            # # Optional debug
            # vals = vec.get_attr("goal_bound")
            # print("\nUpdated bounds:", vals)

    def objective_rew_scale(trial):
        rew_head_scale          = trial.suggest_float("rew_head_scale",     low=5.0,    high=15.0,  step=0.5)
        rew_head_approach_scale = trial.suggest_float("rew_head_app_scale", low= 200,   high=300.0, step=10.0)
        rew_dist_scale          = trial.suggest_float("rew_dist_scale",     low=2.5,    high=5.0,   step=0.5)
        rew_dist_approach_scale = trial.suggest_float("rew_dist_app_scale", low=170.0,  high=250.0, step=10.0)
        rew_time                = trial.suggest_float("rew_time",           low=-0.4,   high=-0.25, step = 0.05)

        for i in range(1,2):
            reward_scale= {
                "rew_head_scale": rew_head_scale,
                "rew_head_approach_scale": rew_head_approach_scale,
                "rew_dist_scale": rew_dist_scale,
                "rew_dist_approach_scale": rew_dist_approach_scale,
                "rew_time": rew_time,
                "rew_goal_scale": 5_000.0,
                "rew_obst_scale": -1_000.0
            }

            # Environment vectorization
            n_proc = 24
        
            print("Making subprocess vectorized environments!")
            env = make_vec_env("Nav2D-v0", 
                                n_envs=n_proc, 
                                env_kwargs={"max_episode_steps": 1_000,
                                            "reward_scale_options": reward_scale
                                            },
                                vec_env_cls=SubprocVecEnv, 
                                vec_env_kwargs=dict(start_method='forkserver'))

            # Hyperparameters
            learning_rate = 1e-4
            buffer_size=int(1e6)
            learning_starts=10_000
            batch_size=4096 
            tau=5e-3
            gamma=0.99
            train_freq=1
            gradient_steps=1
            action_noise=None
            n_steps=1
            policy_delay=2
            target_policy_noise=0.2
            target_noise_clip=0.5
            verbose=0
            dir_path = os.path.dirname(os.path.abspath(__file__))
            tensor_board_log_dir=os.path.join(dir_path,"results","Nav2D_TD3_SB3_optuna_tensorboard")
            print(tensor_board_log_dir)
            pi_arch = [256, 256]
            qf_arch = [256, 256]
            policy_kwargs=dict(activation_fn=torch.nn.ReLU,
                            net_arch=dict(pi=pi_arch, qf=qf_arch))
            use_custom_policy = False
            cuda_enabled = True
            
            # Create the model
            model = TD3("MlpPolicy", env, 
                    learning_rate=learning_rate,         # lr for all networds - Q-values, Actor, Value function
                    buffer_size=buffer_size,         # replay buffer size
                    learning_starts=learning_starts,        # # of data collection step before training
                    batch_size=batch_size,
                    tau=tau,                   # polyak update coefficient
                    gamma=gamma,
                    train_freq=train_freq,
                    gradient_steps=gradient_steps, 
                    action_noise=action_noise, 
                    n_steps=n_steps,                  # n-step TD learning
                    policy_delay=policy_delay,             # the policy and target networks are updated every policy_delay steps
                    target_policy_noise=target_policy_noise,    # stdev of noise added to target policy
                    target_noise_clip=target_noise_clip,      # limit of asbsolute value of noise
                    verbose=verbose,
                    device="cuda" if cuda_enabled else "cpu",
                    policy_kwargs=policy_kwargs if use_custom_policy else None,
                    tensorboard_log=tensor_board_log_dir)
            
            # Training code
            # run parameters:
            number_of_runs = 100
            steps_per_run = 20_000
            model_save_freq = int(number_of_runs / 5)

            # model saving parameters:
            base_path = os.path.join(dir_path, "results", "Nav2D_TD3_SB3_optuna_results")
            result_number = f"result_{len(os.listdir(base_path)):05d}"
            results_path = os.path.join(base_path, result_number)

            # using model.learn approach:
            for run in tqdm(range(1,number_of_runs+1), ncols = 100, colour = "#33FF00", desc = f"{result_number} training progress"):
                # learn every run:
                model.learn(total_timesteps = steps_per_run, tb_log_name=f"{result_number}",reset_num_timesteps = False)
                # model.learn(total_timesteps = steps_per_run, reset_num_timesteps = False)

                # save a model once in a while
                if run % model_save_freq == 0:
                    model.save(os.path.join(results_path, f"run_{run}"))
    
            # save the last model
            model.save(os.path.join(results_path, f"run_{run}"))

            # close environment when done:
            env.close()

            # Save the result-params mapping into a json file
            trial_to_param_path = os.path.join(base_path,'trial_to_param.json')
            if os.path.exists(trial_to_param_path):
                with open(trial_to_param_path, "r") as f:
                    data = json.load(f)
            else:
                data = {result_number: ""}

            hyperparam_codified = f"{learning_rate}_{buffer_size}_{learning_starts}_{batch_size}_{tau}_{gamma}_"
            hyperparam_codified += f"{train_freq}_{gradient_steps}_{n_steps}_{policy_delay}_{target_policy_noise}_{target_noise_clip}_"
            hyperparam_codified += f"{reward_scale['rew_head_scale']}_{reward_scale['rew_head_approach_scale']}_{reward_scale['rew_dist_scale']}_{reward_scale['rew_dist_approach_scale']}_{reward_scale['rew_goal_scale']}_{reward_scale['rew_obst_scale']}"

            timestamp = datetime.now().strftime("%y%m%d_%H%M")
            hyperparam_codified_time = f"{timestamp}_" + hyperparam_codified

            data[result_number] = hyperparam_codified_time

            with open(trial_to_param_path, "w") as f:
                json.dump(data, f, indent=2)

            # Evaluate the policy
            n_evals = 100
            eval_env = gym.make("Nav2D-v0", max_episode_steps = 1_000, render_mode = "rgb_array", is_eval=True)
            mean_eval_rew = eval_policy(env=eval_env, num_evals=n_evals, model=model)
            eval_env.close()

        return mean_eval_rew    # return the mean of evaluation reward as the maximizing objective
    
    def rew_scale_optuna():
        study_name = "rew_scale_nov13"
        study = optuna.create_study(storage=f"sqlite:///python/environments/results/optuna_study.db", 
                                    study_name=study_name, 
                                    load_if_exists=True,
                                    direction='maximize')
        study.optimize(objective_rew_scale, n_trials=100)

    def objective_hyperparam(trial):
        for i in range(1,2):
            reward_scale= {
                "rew_head_scale": 14.5,
                "rew_head_approach_scale": 250.0,
                "rew_dist_scale": 5.0,
                "rew_dist_approach_scale": 190.0,
                "rew_time": -0.25,
                "rew_goal_scale": 5_000.0,
                "rew_obst_scale": -1_000.0
            }

            # Environment vectorization
            n_proc = 24
        
            print("Making subprocess vectorized environments!")
            env = make_vec_env("Nav2D-v0", 
                                n_envs=n_proc, 
                                env_kwargs={"max_episode_steps": 1_000,
                                            "reward_scale_options": reward_scale
                                            },
                                vec_env_cls=SubprocVecEnv, 
                                vec_env_kwargs=dict(start_method='forkserver'))

            # Hyperparameters
            learning_rate   = trial.suggest_categorical(name="lr", choices=[5e-5, 1e-4, 5e-4, 1e-3, 5e-3])
            buffer_size     = trial.suggest_categorical(name="buff_size", choices=[200_000, 500_000, 1_000_000, 2_000_000, 5_000_000])
            learning_starts = trial.suggest_categorical(name="learn_start", choices=[0, 10_000, 50_000])
            batch_size      = trial.suggest_categorical(name="batch", choices=[1024, 2048, 4096])
            tau             = trial.suggest_categorical(name="tau", choices=[5e-4, 1e-3, 5e-3])
            gamma=0.99
            train_freq      = trial.suggest_categorical(name="train_freq", choices=[1, 2, 4])
            gradient_steps  = trial.suggest_categorical(name="grad_step", choices=[1, 2, 4])
            action_noise=None
            n_steps=1
            policy_delay    = trial.suggest_categorical(name="policy_delay", choices=[1, 2, 4])
            target_policy_noise = trial.suggest_float(name="tpolicy_noise", low=0.0, high=0.5, step=0.1)
            target_noise_clip   = trial.suggest_float(name="tnoise_clip", low=0.0, high=1.0, step=0.25)
            verbose=0
            dir_path = os.path.dirname(os.path.abspath(__file__))
            tensor_board_log_dir=os.path.join(dir_path,"results","Nav2D_TD3_SB3_optuna_tensorboard")
            print(tensor_board_log_dir)
            pi_arch = [256, 256]
            qf_arch = [256, 256]
            policy_kwargs=dict(activation_fn=torch.nn.ReLU,
                            net_arch=dict(pi=pi_arch, qf=qf_arch))
            use_custom_policy = False
            cuda_enabled = True
            
            # Create the model
            model = TD3("MlpPolicy", env, 
                    learning_rate=learning_rate,         # lr for all networds - Q-values, Actor, Value function
                    buffer_size=buffer_size,         # replay buffer size
                    learning_starts=learning_starts,        # # of data collection step before training
                    batch_size=batch_size,
                    tau=tau,                   # polyak update coefficient
                    gamma=gamma,
                    train_freq=train_freq,
                    gradient_steps=gradient_steps, 
                    action_noise=action_noise, 
                    n_steps=n_steps,                  # n-step TD learning
                    policy_delay=policy_delay,             # the policy and target networks are updated every policy_delay steps
                    target_policy_noise=target_policy_noise,    # stdev of noise added to target policy
                    target_noise_clip=target_noise_clip,      # limit of asbsolute value of noise
                    verbose=verbose,
                    device="cuda" if cuda_enabled else "cpu",
                    policy_kwargs=policy_kwargs if use_custom_policy else None,
                    tensorboard_log=tensor_board_log_dir)
            
            # Training code
            # run parameters:
            number_of_runs = 100
            steps_per_run = 20_000
            model_save_freq = int(number_of_runs / 5)

            # model saving parameters:
            base_path = os.path.join(dir_path, "results", "Nav2D_TD3_SB3_optuna_results")
            result_number = f"result_{len(os.listdir(base_path)):05d}"
            results_path = os.path.join(base_path, result_number)

            # using model.learn approach:
            for run in tqdm(range(1,number_of_runs+1), ncols = 100, colour = "#33FF00", desc = f"{result_number} training progress"):
                # learn every run:
                model.learn(total_timesteps = steps_per_run, tb_log_name=f"{result_number}",reset_num_timesteps = False)
                # model.learn(total_timesteps = steps_per_run, reset_num_timesteps = False)

                # save a model once in a while
                if run % model_save_freq == 0:
                    model.save(os.path.join(results_path, f"run_{run}"))
    
            # save the last model
            model.save(os.path.join(results_path, f"run_{run}"))

            # close environment when done:
            env.close()

            # Save the result-params mapping into a json file
            trial_to_param_path = os.path.join(base_path,'trial_to_param.json')
            if os.path.exists(trial_to_param_path):
                with open(trial_to_param_path, "r") as f:
                    data = json.load(f)
            else:
                data = {result_number: ""}

            hyperparam_codified = f"{learning_rate}_{buffer_size}_{learning_starts}_{batch_size}_{tau}_{gamma}_"
            hyperparam_codified += f"{train_freq}_{gradient_steps}_{n_steps}_{policy_delay}_{target_policy_noise}_{target_noise_clip}_"
            hyperparam_codified += f"{reward_scale['rew_head_scale']}_{reward_scale['rew_head_approach_scale']}_{reward_scale['rew_dist_scale']}_{reward_scale['rew_dist_approach_scale']}_{reward_scale['rew_goal_scale']}_{reward_scale['rew_obst_scale']}"

            timestamp = datetime.now().strftime("%y%m%d_%H%M")
            hyperparam_codified_time = f"{timestamp}_" + hyperparam_codified

            data[result_number] = hyperparam_codified_time

            with open(trial_to_param_path, "w") as f:
                json.dump(data, f, indent=2)

            # Evaluate the policy
            n_evals = 100
            eval_env = gym.make("Nav2D-v0", max_episode_steps = 1_000, render_mode = "rgb_array", is_eval=True)
            mean_eval_rew = eval_policy(env=eval_env, num_evals=n_evals, model=model)
            eval_env.close()

        return mean_eval_rew    # return the mean of evaluation reward as the maximizing objective
    
    def hyperparam_optuna():
        study_name = "hyperparam_nov16"
        study = optuna.create_study(storage=f"sqlite:///python/environments/results/optuna_study.db", 
                                    study_name=study_name, 
                                    load_if_exists=True,
                                    direction='maximize')
        study.optimize(objective_hyperparam, n_trials=100)

    def objective_rand_freq(trial):
        for i in range(1,2):
            reward_scale= {
                "rew_head_scale": 14.5,
                "rew_head_approach_scale": 250.0,
                "rew_dist_scale": 5.0,
                "rew_dist_approach_scale": 190.0,
                "rew_time": -0.25,
                "rew_goal_scale": 5_000.0,
                "rew_obst_scale": -1_000.0
            }


            randomization_options = {
                "agent_freq": 1,
                "goal_freq": trial.suggest_int(name="goal_freq", low = 1, high = 25),
            }

            # Environment vectorization
            n_proc = 24
        
            print("Making subprocess vectorized environments!")
            env = make_vec_env("Nav2D-v0", 
                                n_envs=n_proc, 
                                env_kwargs={"max_episode_steps": 1_000,
                                            "reward_scale_options": reward_scale,
                                            "randomization_options": randomization_options
                                            },
                                vec_env_cls=SubprocVecEnv, 
                                vec_env_kwargs=dict(start_method='forkserver'))

            # Hyperparameters
            learning_rate = 5e-4
            buffer_size=int(1e6)
            learning_starts=50_000
            batch_size=1024 
            tau=5e-3
            gamma=0.99
            train_freq=2
            gradient_steps=4
            action_noise=None
            n_steps=1
            policy_delay=4
            target_policy_noise=0.1
            target_noise_clip=0.25
            verbose=0
            dir_path = os.path.dirname(os.path.abspath(__file__))
            tensor_board_log_dir=os.path.join(dir_path,"results","Nav2D_TD3_SB3_tensorboard")
            print(tensor_board_log_dir)
            pi_arch = [512, 256]
            qf_arch = [512, 256]
            policy_kwargs=dict(activation_fn=torch.nn.ReLU,
                            net_arch=dict(pi=pi_arch, qf=qf_arch))
            use_custom_policy = False
            cuda_enabled = True
            
            # Create the model
            model = TD3("MlpPolicy", env, 
                    learning_rate=learning_rate,         # lr for all networds - Q-values, Actor, Value function
                    buffer_size=buffer_size,         # replay buffer size
                    learning_starts=learning_starts,        # # of data collection step before training
                    batch_size=batch_size,
                    tau=tau,                   # polyak update coefficient
                    gamma=gamma,
                    train_freq=train_freq,
                    gradient_steps=gradient_steps, 
                    action_noise=action_noise, 
                    n_steps=n_steps,                  # n-step TD learning
                    policy_delay=policy_delay,             # the policy and target networks are updated every policy_delay steps
                    target_policy_noise=target_policy_noise,    # stdev of noise added to target policy
                    target_noise_clip=target_noise_clip,      # limit of asbsolute value of noise
                    verbose=verbose,
                    device="cuda" if cuda_enabled else "cpu",
                    policy_kwargs=policy_kwargs if use_custom_policy else None,
                    tensorboard_log=tensor_board_log_dir)
            
            # Training code
            # run parameters:
            number_of_runs = 100
            steps_per_run = 20_000
            model_save_freq = int(number_of_runs / 5)

            # model saving parameters:
            base_path = os.path.join(dir_path, "results", "Nav2D_TD3_SB3_optuna_results")
            result_number = f"result_{len(os.listdir(base_path)):05d}"
            results_path = os.path.join(base_path, result_number)

            # using model.learn approach:
            for run in tqdm(range(1,number_of_runs+1), ncols = 100, colour = "#33FF00", desc = f"{result_number} training progress"):
                # learn every run:
                model.learn(total_timesteps = steps_per_run, tb_log_name=f"{result_number}",reset_num_timesteps = False)
                # model.learn(total_timesteps = steps_per_run, reset_num_timesteps = False)

                # save a model once in a while
                if run % model_save_freq == 0:
                    model.save(os.path.join(results_path, f"run_{run}"))
    
            # save the last model
            model.save(os.path.join(results_path, f"run_{run}"))

            # close environment when done:
            env.close()

            # Save the result-params mapping into a json file
            trial_to_param_path = os.path.join(base_path,'trial_to_param.json')
            if os.path.exists(trial_to_param_path):
                with open(trial_to_param_path, "r") as f:
                    data = json.load(f)
            else:
                data = {result_number: ""}

            hyperparam_codified = f"{learning_rate}_{buffer_size}_{learning_starts}_{batch_size}_{tau}_{gamma}_"
            hyperparam_codified += f"{train_freq}_{gradient_steps}_{n_steps}_{policy_delay}_{target_policy_noise}_{target_noise_clip}_"
            hyperparam_codified += f"{reward_scale['rew_head_scale']}_{reward_scale['rew_head_approach_scale']}_{reward_scale['rew_dist_scale']}_{reward_scale['rew_dist_approach_scale']}_{reward_scale['rew_goal_scale']}_{reward_scale['rew_obst_scale']}"

            timestamp = datetime.now().strftime("%y%m%d_%H%M")
            hyperparam_codified_time = f"{timestamp}_" + hyperparam_codified

            data[result_number] = hyperparam_codified_time

            with open(trial_to_param_path, "w") as f:
                json.dump(data, f, indent=2)

            # Evaluate the policy
            n_evals = 100
            eval_env = gym.make("Nav2D-v0", max_episode_steps = 1_000, render_mode = "rgb_array", is_eval=True)
            mean_eval_rew = eval_policy(env=eval_env, num_evals=n_evals, model=model)
            eval_env.close()

        return mean_eval_rew    # return the mean of evaluation reward as the maximizing objective
    
    def rand_freq_optuna():
        study_name = "rand_freq_nov17"
        study = optuna.create_study(storage=f"sqlite:///python/environments/results/optuna_study.db", 
                                    study_name=study_name, 
                                    load_if_exists=True,
                                    direction='maximize')
        study.optimize(objective_rand_freq, n_trials=100)

    def train():
        reward_scale= {
                "rew_head_scale": 14.5,
                "rew_head_approach_scale": 250.0,
                "rew_dist_scale": 05.0,
                "rew_dist_approach_scale": 190.0,
                "rew_time": -00.25,
                "rew_goal_scale": 5_000.0,
                "rew_obst_scale": -1_000.0
            }
        
        randomization_options = {
                "agent_freq": 1,
                "goal_freq": 5
            }

        # Environment vectorization
        n_proc = 24

        print("Making subprocess vectorized environments!")
        env = make_vec_env("Nav2D-v0", 
                            n_envs=n_proc, 
                            env_kwargs={"max_episode_steps": 1_000,
                                        "reward_scale_options": reward_scale,
                                        "randomization_options": randomization_options
                                        },
                            vec_env_cls=SubprocVecEnv, 
                            vec_env_kwargs=dict(start_method='forkserver'))
    
        n_actions = env.get_attr("action_space")[0].shape[0]        # obtain the size of the action_space from the list of action_space among the n_proc subprocess envs

        # Hyperparameters
        learning_rate = 5e-4
        buffer_size=int(1e6)
        learning_starts=50_000
        batch_size=1024 
        tau=5e-3
        gamma=0.99
        train_freq=2
        gradient_steps=4
        action_noise=NormalActionNoise(mean=np.zeros(n_actions), sigma=0.03*np.ones(n_actions))     
        n_steps=1
        policy_delay=4
        target_policy_noise=0.1
        target_noise_clip=0.25
        verbose=0
        dir_path = os.path.dirname(os.path.abspath(__file__))
        tensor_board_log_dir=os.path.join(dir_path,"results","Nav2D_TD3_SB3_tensorboard")
        print(tensor_board_log_dir)
        pi_arch = [512, 256]
        qf_arch = [512, 256]
        policy_kwargs=dict(activation_fn=torch.nn.ReLU,
                        net_arch=dict(pi=pi_arch, qf=qf_arch))
        use_custom_policy = False
        cuda_enabled = True

        
        env = VecNormalize(env, training=True, norm_obs=True, norm_reward=True, gamma=gamma)
        
        # Create the model
        model = TD3("MlpPolicy", env, 
                learning_rate=learning_rate,         # lr for all networds - Q-values, Actor, Value function
                buffer_size=buffer_size,         # replay buffer size
                learning_starts=learning_starts,        # # of data collection step before training
                batch_size=batch_size,
                tau=tau,                   # polyak update coefficient
                gamma=gamma,
                train_freq=train_freq,
                gradient_steps=gradient_steps, 
                action_noise=action_noise, 
                n_steps=n_steps,                  # n-step TD learning
                policy_delay=policy_delay,             # the policy and target networks are updated every policy_delay steps
                target_policy_noise=target_policy_noise,    # stdev of noise added to target policy
                target_noise_clip=target_noise_clip,      # limit of asbsolute value of noise
                verbose=verbose,
                device="cuda" if cuda_enabled else "cpu",
                policy_kwargs=policy_kwargs if use_custom_policy else None,
                tensorboard_log=tensor_board_log_dir)
        
        # Training code
        # run parameters:
        number_of_runs = 1_000
        steps_per_run = 50_000
        models_to_save = 10
        model_save_freq = int(number_of_runs / models_to_save)

        # model saving parameters:
        base_path = os.path.join(dir_path, "results", "Nav2D_TD3_SB3_results")
        result_number = f"result_{len(os.listdir(base_path)):05d}"
        results_path = os.path.join(base_path, result_number)

        curriculum_callback = CurriculumCallback(success_window=100, threshold=0.8)

        # using model.learn approach:
        for run in tqdm(range(1,number_of_runs+1), ncols = 100, colour = "#33FF00", desc = f"{result_number} training progress"):
            # learn every run:
            model.learn(total_timesteps = steps_per_run, 
                        tb_log_name=f"{result_number}",
                        reset_num_timesteps = False,
                        callback=curriculum_callback)

            # save a model and the normalization stats once in a while
            if run % model_save_freq == 0:
                model.save(os.path.join(results_path, f"run_{run}"))
                vec_norm_env = model.get_env()
                vec_norm_env.save(os.path.join(results_path, f"norm_stats_{run}.pkl"))
                

        # save the last model
        model.save(os.path.join(results_path, f"run_{run}"))
        vec_norm_env = model.get_env()
        vec_norm_env.save(os.path.join(results_path, f"norm_stats_{run}.pkl"))

        # close environment when done:
        env.close()

        # Save the result-params mapping into a json file
        trial_to_param_path = os.path.join(base_path,'trial_to_param.json')
        if os.path.exists(trial_to_param_path):
            with open(trial_to_param_path, "r") as f:
                data = json.load(f)
        else:
            data = {result_number: ""}

        hyperparam_codified = f"{learning_rate}_{buffer_size}_{learning_starts}_{batch_size}_{tau}_{gamma}_"
        hyperparam_codified += f"{train_freq}_{gradient_steps}_{n_steps}_{policy_delay}_{target_policy_noise}_{target_noise_clip}_"
        hyperparam_codified += f"{reward_scale['rew_head_scale']}_{reward_scale['rew_head_approach_scale']}_{reward_scale['rew_dist_scale']}_{reward_scale['rew_dist_approach_scale']}_{reward_scale['rew_goal_scale']}_{reward_scale['rew_obst_scale']}"

        timestamp = datetime.now().strftime("%y%m%d_%H%M")
        hyperparam_codified_time = f"{timestamp}_" + hyperparam_codified

        data[result_number] = hyperparam_codified_time

        with open(trial_to_param_path, "w") as f:
            json.dump(data, f, indent=2)
    
    # RUN OPTUNA STUDY
    # # optuna study on reward scale
    # rew_scale_optuna()

    # # optuna study on model hyperparameters
    # hyperparam_optuna()

    # # optuna study on goal randomzation frequency
    # rand_freq_optuna()

    # Run one training
    for i in range(1):
        train()

if __name__=="__main__":
    main()