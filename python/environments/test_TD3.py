from stable_baselines3 import TD3,SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

import gymnasium as gym
from gymnasium.wrappers import RescaleAction
import torch
import nav2d        # Have to import the nav2d Python script, else we can't make env
import numpy as np
import os, re, json, time
from datetime import datetime
from tqdm import tqdm

import pyautogui

import seaborn as sns
import matplotlib.pyplot as plt
from collections import deque


result_dir = os.path.join(os.getcwd(),"python","environments")
result_nums = [f"result_000{i}"for i in range(73, 75)]
result_nums = ["result_00015"]
run_num = 500

def eval():
    for result_num in result_nums:
        result_path = os.path.join(result_dir, "results", "Nav2D_TD3_SB3_results", result_num)
        run_path = os.path.join(result_path, f"run_{run_num}")
        
        # testing parameters
        n_test = 50
        report_freq = 50
        success_window = deque(maxlen=report_freq)
        normalized = True

        # environment options
        width = 400
        height = 400
        default_camera_config = {"azimuth" : 90.0, "elevation" : -90.0, "distance" : 3, "lookat" : [0.0, 0.0, 0.0]}
        render_mode = "human" if n_test<=20 else "rgb_array"
        camera_id = 2

        DEFAULT_CAMERA = "overhead_camera"
        ENABLE_FRAME = True                     # enable the body frames
        RENDER_EVERY_FRAME = True              # similar sim speed as MuJoCo rendering when set to False, else slower

        reward_scale= {
                    "rew_head_scale": 2.0,
                    "rew_head_approach_scale": 50.0,
                    "rew_dist_scale": 2.0,
                    "rew_dist_approach_scale": 50.0,
                    "rew_time": -0.1,
                    "rew_goal_scale": 5_000.0,
                    "rew_obst_scale": -10_000.0
                }

        randomization_options = {
                    "agent_freq": 1,
                    "goal_freq": 1
                }
            
        core_env = gym.make("Nav2D-v0", render_mode=render_mode, 
                            width=width,height=height,
                            default_camera_config=default_camera_config,
                            camera_id=camera_id,
                            max_episode_steps=1_000,
                            is_eval=True,
                            reward_scale_options=reward_scale,
                            visual_options = {2: True}
                            )
        if normalized:
            test_env = DummyVecEnv([lambda: core_env])
            # test_env = make_vec_env("Nav2D-v0", 
            #                         n_envs=10, 
            #                         seed=73,
            #                         env_kwargs={"max_episode_steps": 1_000,
            #                                     "reward_scale_options": reward_scale,
            #                                     "randomization_options": randomization_options
            #                                     },
            #                         vec_env_cls=SubprocVecEnv,
            #                         vec_env_kwargs=dict(start_method='forkserver'))
            test_env = VecNormalize.load(os.path.join(result_path, f"norm_stats_{run_num}.pkl"), test_env)
            test_env.training = False
            test_env.norm_reward = False
            test_env.venv.env_method("_set_goal_bound", 0.0)
            
            print(f"created a normalized environment with the goal bound set to {test_env.venv.get_attr('goal_bound')[0]}")
            model_load = TD3.load(run_path, env=test_env)
        else:
            print("created a core environment (unnormalized)")
            model_load = TD3.load(run_path, env=core_env)

        
        if normalized:  # using normalized vectorized environments
            n_envs = test_env.num_envs
            obss = test_env.reset()
            eps_evaluated = 0 

            while eps_evaluated < n_test:
                action, _ = model_load.predict(obss, deterministic=True)
                nobss, rews, dones, infos = test_env.step(action)

                for i in range(n_envs):
                    if dones[i]:
                        eps_evaluated += 1

                        info = infos[i]
                        success_window.append(info.get("is_success", False))
                        
                        # reset the current env
                        result = test_env.env_method("reset", indices=i)[0]
                        if isinstance(result,tuple):        # if reset() return obs and info
                            nobss[i], _ = result    
                        else:                               # if reset() only return obs
                            nobss[i] = result
                obss = nobss

                if len(success_window) >= report_freq:
                    print(f"Success rate from run {eps_evaluated-report_freq+1} to {eps_evaluated} is {sum(success_window)/len(success_window)*100:.2f}%             ")
                    success_window.clear()
                        
        else:           # using core gymnasium environment
            for eps_evaluated in tqdm(range(1,n_test+1), ncols = 100, colour = "#33FF00", desc = f"Evaluating {result_num}..."):
                done = False
                obs, info = test_env.reset()
                acc_rew = 0
                step_count = 0

                while not done:
                    action, _ = model_load.predict(obs, deterministic=True)
                    nobs, rew, term, trunc, info = test_env.step(action)
                    done = term or trunc
                    acc_rew += rew
                    step_count += 1

                    if not done: obs = nobs
                    else:
                        success_window.append(info.get("is_success", False))

                if len(success_window) >= report_freq:
                    print(f"Success rate from run {eps_evaluated-report_freq+1} to {eps_evaluated} is {sum(success_window)/len(success_window)*100:.2f}%             ")
                    success_window.clear()
        test_env.close()

if __name__=="__main__":
    eval()