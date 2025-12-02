# ===================================================================
# This script is used for training TD3 using stable baselines 3 with 
# SubprocVecEnv for multi-core processing, with either the "forkserver" 
# or the "spawn" method

# These methods only work when the training code is wrapped in a 
# if __name__ == "main" block
# ===================================================================

import stable_baselines3
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

        # unwrap the environment to obtain maximum episode length and arena size
        if isinstance(env, (DummyVecEnv, SubprocVecEnv, VecNormalize)):
            max_eps_length = env.envs[0].spec.max_episode_steps
            size = env.envs[0].unwrapped.size
            max_env_size = np.sqrt(2* (2 * size)**2)
            is_vec_env = True
            obs = env.reset()
        else:    
            max_eps_length = env.spec.max_episode_steps
            max_env_size = np.sqrt(2* (2 * env.unwrapped.size)**2)
            obs, _ = env.reset()
            is_vec_env = False
        
        # empty list to later store the results
        success_list = [0]*num_evals
        ep_len_list = [0]*num_evals
        final_dist_list = [0]*num_evals
        
        # evaluate the model
        for ep in range(num_evals):
            done = False

            # initialize episodic reward:
            ep_len = 0

            # while False:
            while not done:
                # get action and step:
                with torch.no_grad():
                    action, _ = model.predict(obs, deterministic = True)

                    if is_vec_env:
                        nobs, rew, done, info = env.step(action)
                    else:
                        nobs, reward, term, trunc, info = env.step(action)
                        done = term or trunc
                    
                    ep_len += 1

                    # advance observation, reset if not:
                    if done:
                        success_list[ep] = info[0].get("is_success",False) if is_vec_env else info.get("is_success",False)
                        ep_len_list[ep] = ep_len
                        final_dist_list[ep] = np.sqrt(nobs[0][1]**2+nobs[0][1]**2) if is_vec_env else np.sqrt(nobs[0]**2+nobs[1]**2)
                        obs = env.reset() if is_vec_env else env.reset()[0]
                    else:
                        obs = nobs

        mean_success = np.mean(np.mean(success_list))
        mean_ep_len = np.round(np.mean(ep_len_list)/max_eps_length,3)
        mean_final_dist = np.round(np.mean(final_dist_list)/max_env_size,3)

        return mean_success, mean_ep_len, mean_final_dist

    class CurriculumCallback(BaseCallback):
        def __init__(self, success_window = 100, threshold = 0.9, cooldown_eps = 200):
            super().__init__()
            self.success_threshold = threshold
            self.success_window = success_window
            self.buffer = deque(maxlen=success_window)
            self.goal_bound_levels = np.linspace(0,1,11)
            self.goal_level_idx = 0
            self.agent_bound_levels = np.linspace(0,1,5)
            self.agent_level_idx = 0

            self.explore_boost_steps = 0            # keep count of the number of steps with increased exploration (higher action noise)

            self.cooldown_eps = cooldown_eps
            self.cooldown_active = False
            self.cooldown_remaining = 0

        def _on_step(self) -> bool:
            # grab info dict from each parallel env
            infos = self.locals["infos"]

            for info in infos:
                if "is_success" in info:        # check terminal state in each env
                    self.buffer.append(info["is_success"])
                    if self.cooldown_active: self.cooldown_remaining -= 1
                elif info.get("TimeLimit.truncated", False):                           # in case of truncation, there is no "is_success" in info => unsuccessful
                    self.buffer.append(False)
                    if self.cooldown_active: self.cooldown_remaining -= 1

            # cooldown episode countdown after curriculum advancement
            if self.cooldown_active and (self.cooldown_remaining <= 0): 
                self.cooldown_active = False
                self.cooldown_remaining = 0

            # curriculum advancement logic
            if (not self.cooldown_active 
                and len(self.buffer) >= self.success_window):

                success_rate = sum(self.buffer)/len(self.buffer)
                print(f"Current training success {success_rate:3.2f}", end="\r")

                # if the success rate of the past N episodes exceed success threshold:
                # 1. advance the agent curriculum
                # 2. if the agent is randomized at maximum bound, start advancing the goal curriculum 
                if (success_rate > self.success_threshold) :
                    if self.agent_level_idx < len(self.agent_bound_levels) - 1:
                        self._advance_agent_curriculum()
                        self.buffer.clear()
                        self._start_cooldown()
                    else:
                        self._advance_goal_curriculum()
                        self.buffer.clear()
                        self._start_cooldown()
                        pass

            # countdown of episodes with boosted exploration
            if self.explore_boost_steps > 0:
                self.explore_boost_steps -= 1
                if self.explore_boost_steps == 0:
                    
                    vec_noise = self.model.action_noise
                    for i in vec_noise.noises:
                        i._sigma = self.noise_std_backup.copy()
                    
                    print(f"\nFinished with steps of increased exploration, noise reset to {i._sigma} \n")
            return True     # to continue training, return True, else return False
        
        def _start_cooldown(self):
            '''Function to countd down the episodes of no curriculum advancing right after an advancement'''
            self.cooldown_active = True
            self.cooldown_remaining = self.cooldown_eps
            print(f"\nStarting cooldown for {self.cooldown_eps} terminations.\n")

        def _boost_explore(self, num_steps):
            self.explore_boost_steps = num_steps
            vec_noise = self.model.action_noise             # VectorizedActionNoise, which is a vector of the action noise (NormalActionNoise)
            noise = vec_noise.noises                        # list of NormalActionNoise instance of one environment
            self.noise_std_backup = noise[0]._sigma.copy()  # keep a copy of the original noise stdev

            for i in noise:
                i._sigma = [x+0.1 for x in i._sigma]     # increase action noise stdev for all child envs
            print(f"Action noise => {i._sigma} for {self.explore_boost_steps} steps")

        def _advance_agent_curriculum(self):
            ''' Function to advance the agent curriculum by
            1. inncreasing the agent randomization bound
            2. boost the action_noise in each environment to encourage exploratory actions'''

            #--- increase the agent bound
            env = self.model.get_env()  # vecnormalized wrapped vec env
            vec = env.venv              # unwrap to vec env
            self.agent_level_idx += 1
            self.agent_level_idx = min(self.agent_level_idx, len(self.agent_bound_levels) - 1)

            vec.env_method("_set_agent_bound", self.agent_bound_levels[self.agent_level_idx])
            current_agent_bound = vec.get_attr("agent_bound")[0]      # get the actual goal_bound from one environment
            print(f"\nIncreasing agent bound - level {self.agent_level_idx:2d} | goal_bound = {current_agent_bound:4.3f} | success_window = {self.success_window}")

            #--- boost exploration for several steps
            self._boost_explore(num_steps=1_000)

        def _advance_goal_curriculum(self):
            ''' Function to advance the curriculum by
            1. increasing the goal randomization bound 
            2. boost action_noise to increase exploration for a set number of steps
            '''
            #--- increase the goal bound
            env = self.model.get_env()  # vecnormalized wrapped vec env
            vec = env.venv              # unwrap to vec env
            self.goal_level_idx += 1
            self.goal_level_idx = min(self.goal_level_idx, len(self.goal_bound_levels) - 1)

            vec.env_method("_set_goal_bound", self.goal_bound_levels[self.goal_level_idx])
            current_goal_bound = vec.get_attr("goal_bound")[0]      # get the actual goal_bound from one environment
            print(f"\nIncreasing goal bound - level {self.goal_level_idx:2d} | goal_bound = {current_goal_bound:4.3f} | success_window = {self.success_window}")

            #--- boost exploration for several steps
            self._boost_explore(num_steps=2_000)

    def objective_rew_scale(trial):
        # w_head          = trial.suggest_float("w_head",     low=0.5,    high=3.0,  step=0.25)
        # w_head_app      = trial.suggest_float("w_head_app", low=10.0,   high=50.0,  step=10.0)
        # w_dist          = trial.suggest_float("w_dist",     low=0.5,    high=3.0,  step=0.25)
        # w_dist_app      = trial.suggest_float("w_dist_app", low=10.0,   high=50.0,  step=10.0)
        # rew_time        = trial.suggest_float("rew_time",   low=-0.25,  high=-0.05, step=0.05)

        w_head      = trial.suggest_float       (name="w_head",     low=0.5,    high=3.0,  step=0.25)
        hd_ratio    = trial.suggest_categorical (name="hd_ratio",   choices = [0.5, 1.0, 2.0])
        ps_ratio    = trial.suggest_categorical (name="ps_ratio",   choices = [0.5, 1.0, 2.0])
        w_goal      = trial.suggest_categorical (name="w_goal",     choices = [500.0, 5000.0, 1000.0])
        go_ratio    = trial.suggest_categorical (name="go_ratio",   choices = [0.5, 1.0, 2.0])

        w_dist      = w_head * hd_ratio
        w_dist_app  = w_dist * ps_ratio
        w_head_app  = w_head * ps_ratio
        w_obstacle  = - w_goal * go_ratio
        
        reward_scale= {
            "rew_head_scale":           w_head,
            "rew_head_approach_scale":  w_head_app,
            "rew_dist_scale":           w_dist,
            "rew_dist_approach_scale":  w_dist_app,
            "rew_time":                 -0.1,
            "rew_goal_scale": w_goal,
            "rew_obst_scale": w_obstacle
        }

        # Environment vectorization & normalization
        n_proc = 25
        normalize = True
    
        print("Making subprocess vectorized environments!")
        env = make_vec_env("Nav2D-v0", 
                            n_envs=n_proc, 
                            env_kwargs={"max_episode_steps": 1_000,
                                        "reward_scale_options": reward_scale
                                        },
                            vec_env_cls=SubprocVecEnv, 
                            vec_env_kwargs=dict(start_method='forkserver'))
        
        n_actions = env.get_attr("action_space")[0].shape[0]        # obtain the size of the action_space from the list of action_space among the n_proc subprocess envs

        # Hyperparameters
        learning_rate = 1e-3
        buffer_size=int(1e6)
        learning_starts=50_000
        batch_size=1024
        tau=5e-3
        gamma=0.99
        train_freq=2
        gradient_steps=4
        act_noise_std=0.03
        action_noise=NormalActionNoise(mean=np.zeros(n_actions), sigma=act_noise_std*np.ones(n_actions))     
        n_steps=1
        policy_delay=4
        target_policy_noise=0.1
        target_noise_clip=0.25
        verbose=0
        dir_path = os.path.dirname(os.path.abspath(__file__))
        tensor_board_log_dir=os.path.join(dir_path,"results","Nav2D_TD3_SB3_optuna_tensorboard")
        print(tensor_board_log_dir)
        pi_arch = [512, 256]
        qf_arch = [512, 256]
        policy_kwargs=dict(activation_fn=torch.nn.ReLU,
                        net_arch=dict(pi=pi_arch, qf=qf_arch))
        use_custom_policy = False
        cuda_enabled = True

        normalize = True
        if normalize:
            env = VecNormalize(env, training=True, norm_obs=True, norm_reward=True, gamma=gamma)
        env.reset()
        
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
        models_to_save = 5
        model_save_freq = int(number_of_runs / models_to_save)

        # model saving parameters:
        base_path = os.path.join(dir_path, "results", "Nav2D_TD3_SB3_optuna_results")
        result_number = f"result_{len(os.listdir(base_path))-1:05d}"
        results_path = os.path.join(base_path, result_number)

        # Save the result-params mapping into a json file
        trial_to_param_path = os.path.join(base_path,'trial_to_param.json')
        if os.path.exists(trial_to_param_path):
            with open(trial_to_param_path, "r") as f:
                data = json.load(f)
        else:
            data = {result_number: ""}

        hyperparam_codified = f"{learning_rate}_{buffer_size}_{learning_starts}_{batch_size}_{tau}_{gamma}_"
        hyperparam_codified += f"{train_freq}_{gradient_steps}_{act_noise_std}_{n_steps}_{policy_delay}_{target_policy_noise}_{target_noise_clip}_"
        hyperparam_codified += f"{reward_scale['rew_head_scale']}_{reward_scale['rew_head_approach_scale']}_{reward_scale['rew_dist_scale']}_{reward_scale['rew_dist_approach_scale']}_{reward_scale['rew_goal_scale']}_{reward_scale['rew_obst_scale']}"

        timestamp = datetime.now().strftime("%y%m%d_%H%M")
        hyperparam_codified_time = f"{timestamp}_" + hyperparam_codified

        data[result_number] = hyperparam_codified_time

        with open(trial_to_param_path, "w") as f:
            json.dump(data, f, indent=2)

        # training
        for run in tqdm(range(1,number_of_runs+1), ncols = 100, colour = "#33FF00", desc = f"{result_number} training progress"):
            # learn every run:
            model.learn(total_timesteps = steps_per_run, 
                        tb_log_name=f"{result_number}",
                        reset_num_timesteps = False,
                        # callback=curriculum_callback
                        )

            # save a model and the normalization stats once in a while
            if run % model_save_freq == 0:
                model.save(os.path.join(results_path, f"run_{run}"))
                
                if normalize:
                    vec_norm_env = model.get_env()
                    vec_norm_env.save(os.path.join(results_path, f"norm_stats_{run}.pkl"))
                

        # save the last model
        model.save(os.path.join(results_path, f"run_{run}"))

        if normalize:
            vec_norm_env = model.get_env()
            vec_norm_env.save(os.path.join(results_path, f"norm_stats_{run}.pkl"))

        # close environment when done:
        env.close()

        # Evaluate the policy
        n_evals = 100
        eval_env = gym.make("Nav2D-v0", max_episode_steps = 1_000, render_mode = "rgb_array", is_eval=True)
        if normalize:
            eval_env = DummyVecEnv([lambda: eval_env])
            eval_env = VecNormalize.load(os.path.join(results_path, f"norm_stats_{run}.pkl"), eval_env)
            eval_env.training = False
            eval_env.norm_reward = False
            print("created a normalized environment!")
            obs = eval_env.reset()
        else:
            obs, info = eval_env.reset()
        success, ep_len, final_dist = eval_policy(env=eval_env, num_evals=n_evals, model=model)
        eval_env.close()

        return success, ep_len, final_dist
    
    def rew_scale_optuna():
        study_name = "rew_scale_nov24"
        study = optuna.create_study(storage=f"sqlite:///python/environments/results/optuna_study.db", 
                                    study_name=study_name, 
                                    load_if_exists=True,
                                    directions=['maximize', 'minimize', "minimize"])
        study.set_metric_names(["success_rate","ep_len","final_dist"])
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
            n_proc = 25
        
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
                "agent_freq": trial.suggest_int(name="agent_freq", low = 1, high = 5, step = 1),
                "goal_freq": trial.suggest_int(name="goal_freq", low = 5, high = 50, step = 5),
            }

            # Environment vectorization
            n_proc = 24

            # Hyperparameters
            learning_rate = 1e-3
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
            tensor_board_log_dir=os.path.join(dir_path,"results","Nav2D_TD3_SB3_optuna_tensorboard")
            print(tensor_board_log_dir)
            pi_arch = [512, 256]
            qf_arch = [512, 256]
            policy_kwargs=dict(activation_fn=torch.nn.ReLU,
                            net_arch=dict(pi=pi_arch, qf=qf_arch))
            use_custom_policy = False
            cuda_enabled = True

            print("Making subprocess vectorized environments!")
            env = make_vec_env("Nav2D-v0", 
                                n_envs=n_proc, 
                                env_kwargs={"max_episode_steps": 1_000,
                                            "reward_scale_options": reward_scale,
                                            "randomization_options": randomization_options
                                            },
                                vec_env_cls=SubprocVecEnv, 
                                vec_env_kwargs=dict(start_method='forkserver'))
            
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
            number_of_runs = 100
            steps_per_run = 50_000
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

            # Evaluate the policy
            n_evals = 100
            eval_env = gym.make("Nav2D-v0", max_episode_steps = 1_000, render_mode = "rgb_array", is_eval=True)
            mean_eval_rew = eval_policy(env=eval_env, num_evals=n_evals, model=model)
            eval_env.close()

        return mean_eval_rew    # return the mean of evaluation reward as the maximizing objective
    
    def rand_freq_optuna():
        study_name = "rand_freq_nov18"
        study = optuna.create_study(storage=f"sqlite:///python/environments/results/optuna_study.db", 
                                    study_name=study_name, 
                                    load_if_exists=True,
                                    direction='maximize')
        study.optimize(objective_rand_freq, n_trials=100)

    def train():
        reward_scale= {
                "rew_head_scale": 0.5,
                "rew_head_approach_scale": 100.0,
                "rew_dist_scale": 1.0,
                "rew_dist_approach_scale": 250.0,
                "rew_time": -0.25,
                "rew_goal_scale": 5_000.0,
                "rew_obst_scale": -1_000.0
            }
        
        randomization_options = {
                "agent_freq": 1,
                "goal_freq": 1
            }

        # Environment vectorization
        n_proc = 24

        print("Making subprocess vectorized environments!")
        env = make_vec_env("Nav2D-v0", 
                            n_envs=n_proc, 
                            seed=73,
                            env_kwargs={"max_episode_steps": 1_000,
                                        "reward_scale_options": reward_scale,
                                        "randomization_options": randomization_options
                                        },
                            vec_env_cls=SubprocVecEnv, 
                            vec_env_kwargs=dict(start_method='forkserver'))
    
        n_actions = env.get_attr("action_space")[0].shape[0]        # obtain the size of the action_space from the list of action_space among the n_proc subprocess envs

        # Hyperparameters
        learning_rate=1e-3
        buffer_size=int(1e6)
        learning_starts=50_000
        batch_size=1024
        tau=5e-3
        gamma=0.99
        train_freq=2
        gradient_steps=4
        act_noise_std=0.0
        action_noise=NormalActionNoise(mean=np.zeros(n_actions), sigma=act_noise_std*np.ones(n_actions))     
        n_steps=1
        policy_delay=4
        target_policy_noise=0.05
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
        
        env = VecNormalize(env, training=True, norm_obs=False, norm_reward=True, gamma=gamma)
        env.reset()
        
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
        models_to_save = 20
        model_save_freq = int(number_of_runs / models_to_save)

        # model saving parameters:
        base_path = os.path.join(dir_path, "results", "Nav2D_TD3_SB3_results")
        result_number = f"result_{len(os.listdir(base_path))-1:05d}"
        results_path = os.path.join(base_path, result_number)

        curriculum_callback = CurriculumCallback(success_window=200, threshold=0.75, cooldown_eps=500)

        # Save the result-params mapping into a json file
        trial_to_param_path = os.path.join(base_path,'trial_to_param.json')
        if os.path.exists(trial_to_param_path):
            with open(trial_to_param_path, "r") as f:
                data = json.load(f)
        else:
            data = {result_number: ""}

        hyperparam_codified = f"{learning_rate}_{buffer_size}_{learning_starts}_{batch_size}_{tau}_{gamma}_"
        hyperparam_codified += f"{train_freq}_{gradient_steps}_{act_noise_std}_{n_steps}_{policy_delay}_{target_policy_noise}_{target_noise_clip}_"
        hyperparam_codified += f"{reward_scale['rew_head_scale']}_{reward_scale['rew_head_approach_scale']}_{reward_scale['rew_dist_scale']}_{reward_scale['rew_dist_approach_scale']}_{reward_scale['rew_goal_scale']}_{reward_scale['rew_obst_scale']}"

        timestamp = datetime.now().strftime("%y%m%d_%H%M")
        hyperparam_codified_time = f"{timestamp}_" + hyperparam_codified

        data[result_number] = hyperparam_codified_time

        with open(trial_to_param_path, "w") as f:
            json.dump(data, f, indent=2)

        # training
        for run in tqdm(range(1,number_of_runs+1), ncols = 100, colour = "#33FF00", desc = f"{result_number} training progress"):
            # learn every run:
            model.learn(total_timesteps = steps_per_run, 
                        tb_log_name=f"{result_number}",
                        reset_num_timesteps = False,
                        callback=curriculum_callback
                        )

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
    
    # # RUN OPTUNA STUDY
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