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
result_nums = ["result_00115"]
run_num = 500
goal_bound_ratio = 1.0

def eval():
    print(f"Evaluating {result_nums}, run {run_num:3d}")
    for result_num in result_nums:
        result_path = os.path.join(result_dir, "results", "Nav2D_TD3_SB3_results", result_num)
        run_path = os.path.join(result_path, f"run_{run_num}")
        
        # testing parameters
        n_test = 50
        report_freq = 100
        success_window = deque(maxlen=report_freq)
        normalized = True

        # environment options
        width = 800
        height = 800
        render_mode = "human" if n_test<=50 else "rgb_array"
        camera_id = 2

        DEFAULT_CAMERA = "overhead_camera"
        ENABLE_FRAME = True                     # enable the body frames
        RENDER_EVERY_FRAME = True              # similar sim speed as MuJoCo rendering when set to False, else slower

        reward_scale= {
            "rew_dist_scale":           2.0,
            "rew_dist_approach_scale":  25.0,
            "rew_head_scale":           1.0,
            "rew_head_approach_scale":  25.0,
            "rew_time":                 -0.1,
            "rew_goal_scale":           10_000.0,
            "rew_obst_scale":           -2_000.0
        }
        
        randomization_options = {
            "agent_freq": 1,
            "goal_freq": 1,
            "obstacle_freq": 1
        }
        term_cond = {
            "is_success": 0,
            "obstacle_cond": 0,
            "dist_progress_cond": 0,
            "head_progress_cond": 0,
            "TimeLimit.truncated": 0
        }
            
        core_env = gym.make("Nav2D-v0", render_mode=render_mode, 
                            width=width,height=height,
                            default_camera_config=default_camera_config,
                            camera_id=camera_id,
                            max_episode_steps=2_500,
                            is_eval=True,
                            reward_scale_options=reward_scale,
                            randomization_options=randomization_options,
                            visual_options = {2: True, 8: True}
                            )
        if normalized:
            if render_mode == "human":
                test_env = DummyVecEnv([lambda: core_env])
            else:
                test_env = make_vec_env("Nav2D-v0", 
                                        n_envs=20,
                                        env_kwargs={"max_episode_steps": 2_500,
                                                    "is_eval": True,
                                                    "reward_scale_options": reward_scale,
                                                    "randomization_options": randomization_options
                                                    },
                                        vec_env_cls=SubprocVecEnv,
                                        vec_env_kwargs=dict(start_method='forkserver'))
            test_env = VecNormalize.load(os.path.join(result_path, f"norm_stats_{run_num}.pkl"), test_env)
            test_env.training = False
            test_env.norm_reward = False

            test_env.venv.env_method("_set_goal_bound", goal_bound_ratio)
            print(f"created a normalized environment with the goal bound set to {test_env.venv.get_attr('goal_bound')[0]} ({goal_bound_ratio*100:5.2f}%)")
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

                # Check for terminal episodes among all the vec_envs
                for i, (done, info) in enumerate(zip(dones, infos)):
                    if done:
                        eps_evaluated += 1
                        for k in term_cond.keys(): term_cond[k] += int(info.get(k,False))
                        success_window.append(info.get("is_success", False))
                        
                obss = nobss

                if len(success_window) >= report_freq:
                    print(f"Success rate from run {eps_evaluated-report_freq+1} to {eps_evaluated} is {sum(success_window)/len(success_window)*100:.2f}%             ")
                    print(f"Termination conditions: {term_cond}")
                    success_window.clear()
                    term_cond = dict.fromkeys(term_cond, 0)
                        
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