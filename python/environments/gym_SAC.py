# ===================================================================
# This script is used for training SAC using stable baselines 3 with 
# SubprocVecEnv for multi-core processing, with either the "forkserver" 
# or the "spawn" method

# These methods only work when the training code is wrapped in a 
# if __name__ == "main" block
# ===================================================================
# imports:
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import torch

import gymnasium as gym
import nav2d
import numpy as np
import os, json
from datetime import datetime
from tqdm import tqdm

# Ignore User Warnings (for creating a new folder to save policies):
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    # loop for scaling study:
        # define reward scaling:
        reward_scale = {
            "rew_head_scale" : 2.5,
            "rew_head_approach_scale" : 50.0,
            "rew_dist_scale" : 250,
            "rew_goal_scale" : 5000.0,
            "rew_obst_scale" : -1000.0
        }

        # environment vectorization settings:
        n_proc = 24

        # episode steps:
        max_episode_steps = 1000

        # create the vectorized environments:
        env = make_vec_env("Nav2D-v0", 
                            n_envs = n_proc,
                            env_kwargs = {"max_episode_steps" : max_episode_steps,
                                        "reward_scale_options" : reward_scale},
                            vec_env_cls = SubprocVecEnv,
                            vec_env_kwargs = dict(start_method = 'spawn'))
        
        # model hyperparameters:
        policy = "MlpPolicy"
        gamma = 0.99
        actor_lr = 1e-4
        critic_lr = 1e-5
        buffer_size = int(1e6)
        batch_size = 4096
        tau = 5e-3
        ent_coef = "auto_0.1"
        train_freq = 1
        learning_starts = 0
        target_update_interval = 1
        gradient_steps = 4
        target_entropy = -env.action_space.shape[0]
        action_noise = None
        verbose = 0
        gpu = True

        # for tensorboard logging:
        dir_path = os.path.dirname(os.path.abspath(__file__))
        tensorboard_log_dir = os.path.join(dir_path, "results", "Nav2D_SAC_SB3_tensorboard")

        # create the model:
        model = SAC(policy = policy,
                    env = env, 
                    gamma = gamma, 
                    buffer_size = buffer_size, 
                    batch_size = batch_size,
                    tau = tau,
                    ent_coef = ent_coef,
                    train_freq = train_freq,
                    learning_starts = learning_starts,
                    target_update_interval = target_update_interval,
                    gradient_steps = gradient_steps,
                    target_entropy = target_entropy,
                    action_noise = action_noise,
                    verbose = verbose,
                    device = "cuda" if gpu else "cpu", 
                    tensorboard_log = tensorboard_log_dir)
        
        # set custom learning rates:
        model.actor.optimizer = torch.optim.Adam(model.actor.parameters(), lr = actor_lr)
        model.critic.optimizer = torch.optim.Adam(model.critic.parameters(), lr = critic_lr)

        # training parameters:
        number_of_runs = 100
        steps_per_run = 50000
        model_save_freq = max(int(number_of_runs / 20), 1)

        # model saving parameters:
        base_path = os.path.join(dir_path, "results", "Nav2D_SAC_SB3_results")
        result_number = f"result_{len(os.listdir(base_path)):03d}"
        results_path = os.path.join(base_path, result_number)

        # train using model.learn():
        for run in tqdm(range(1, number_of_runs + 1), ncols = 100, colour = "#33FF00", desc = f"{result_number} training progress"):
            # learn every run:
            model.learn(total_timesteps = steps_per_run, tb_log_name = f"{result_number}", reset_num_timesteps = False)

            # save a model every now and then:
            if run % model_save_freq == 0:
                model.save(os.path.join(results_path, f"run_{run}"))

        # need to save the last model:
        model.save(os.path.join(results_path, f"run_{run}"))

        # close environment when done:
        env.close()

        # save the result-params mapping into a json file (minh formulation):
        trial_to_param_path = os.path.join(base_path,'trial_to_param.json')
        if os.path.exists(trial_to_param_path):
            with open(trial_to_param_path, "r") as f:
                data = json.load(f)
        else:
            data = {result_number: ""}

        hyperparam_codified = f"{actor_lr}_{critic_lr}_{buffer_size}_{batch_size}_{tau}_{gamma}_"
        hyperparam_codified += f"{train_freq}_{gradient_steps}_{ent_coef}_{target_update_interval}_{target_entropy}_"
        hyperparam_codified += f"{reward_scale['rew_head_scale']}_{reward_scale["rew_head_approach_scale"]}_{reward_scale['rew_dist_scale']}_{reward_scale['rew_goal_scale']}_{reward_scale['rew_obst_scale']}"

        timestamp = datetime.now().strftime("%y%m%d_%H%M")
        hyperparam_codified_time = f"{timestamp}_" + hyperparam_codified

        data[result_number] = hyperparam_codified_time

        with open(trial_to_param_path, "w") as f:
            json.dump(data, f, indent=2)

if __name__=="__main__":
    main()
