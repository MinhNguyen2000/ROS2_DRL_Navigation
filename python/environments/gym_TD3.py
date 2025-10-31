# ===================================================================
# This script is used for training TD3 using stable baselines 3 with 
# SubprocVecEnv for multi-core processing, with either the "forkserver" 
# or the "spawn" method

# These methods only work when the training code is wrapped in a 
# if __name__ == "main" block
# ===================================================================

from stable_baselines3 import TD3,SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

import torch
import gymnasium as gym
from gymnasium.wrappers import RescaleAction
import nav2d        # Have to import the nav2d Python script, else we can't make env
import nav2d_testing
import numpy as np
import os, re, json, time
from datetime import datetime
from tqdm import tqdm

# Ignore User Warnings (for creating a new folder to save policies)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def main():
    for i in range(1,5):
        reward_scale= {
            "rew_head_scale": 5.0,
            "rew_head_approach_scale": 50.0*i,
            "rew_dist_scale": 2_000.0,
            "rew_goal_scale": 2_000.0,
            "rew_obst_scale": -1_000.0
        }

        # Environment vectorization
        n_envs = 4
        n_proc = 24
        dummy_env = False

        # Create the vectorized environments
        if dummy_env:
            print("Making dummy vectorized environments!")
            env = make_vec_env("Nav2D-v0", 
                            n_envs=n_envs,
                            env_kwargs={"max_episode_steps": 1_000,
                                        "reward_scale_options": reward_scale
                                        },
                            vec_env_cls=DummyVecEnv)
        else:
            print("Making subprocess vectorized environments!")
            env = make_vec_env("Nav2D-v0", 
                                n_envs=n_proc, 
                                env_kwargs={"max_episode_steps": 1_000,
                                            "reward_scale_options": reward_scale
                                            },
                                vec_env_cls=SubprocVecEnv, 
                                vec_env_kwargs=dict(start_method='forkserver'))

        # Hyperparameters
        learning_rate = 3e-4
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
        target_policy_noise=0.05
        target_noise_clip=0.1
        verbose=0
        dir_path = os.path.dirname(os.path.abspath(__file__))
        tensor_board_log_dir=os.path.join(dir_path,"results","Nav2D_TD3_SB3_tensorboard")
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
        model_save_freq = int(number_of_runs / 20)

        # model saving parameters:
        base_path = os.path.join(dir_path, "results", "Nav2D_TD3_SB3_results")
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
        hyperparam_codified += f"{reward_scale['rew_head_scale']}_{reward_scale['rew_head_approach_scale']}_{reward_scale['rew_dist_scale']}_{reward_scale['rew_goal_scale']}_{reward_scale['rew_obst_scale']}"

        timestamp = datetime.now().strftime("%y%m%d_%H%M")
        hyperparam_codified_time = f"{timestamp}_" + hyperparam_codified

        data[result_number] = hyperparam_codified_time

        with open(trial_to_param_path, "w") as f:
            json.dump(data, f, indent=2)

if __name__=="__main__":
    main()