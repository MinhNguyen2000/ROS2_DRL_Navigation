from stable_baselines3 import TD3,SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
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


result_dir = os.path.join(os.getcwd(),"python","environments")
result_nums = [f"result_000{i}"for i in range(68,70)]
# result_nums = ["result_00068"]
run_num = "run_100" 

for result_num in result_nums:
    result_path = os.path.join(result_dir, "results", "Nav2D_TD3_SB3_results", result_num)
    run_path = os.path.join(result_path, run_num)
    model_load = TD3.load(run_path)

    # testing parameters
    n_test = 500
    plot_freq = 50
    success_count = 0

    # environment options
    width = 400
    height = 400
    default_camera_config = {"azimuth" : 90.0, "elevation" : -90.0, "distance" : 3, "lookat" : [0.0, 0.0, 0.0]}
    render_mode = "human" if n_test<=20 else "rgb_array"
    camera_id = 2

    DEFAULT_CAMERA = "overhead_camera"
    ENABLE_FRAME = True                     # enable the body frames
    RENDER_EVERY_FRAME = True              # similar sim speed as MuJoCo rendering when set to False, else slower

    test_env = gym.make("Nav2D-v0", render_mode=render_mode, 
                        width=width,height=height,
                        default_camera_config=default_camera_config,
                        camera_id=camera_id,
                        max_episode_steps=1_500,
                        is_eval=False,
                        visual_options = {2: True}
                        )
    obs, info = test_env.reset()

    core_env = test_env.unwrapped
    rew_goal = core_env.rew_goal_scale

    agent_init_list = []                                        # store the initial state of the agent
    rew_head_list = []                                          # store the final heading reward of the agent to quantify how well-pose
    rew_hist_normalized = np.zeros(n_test, dtype=np.float32)    # store the accumulated rewards normalized by the length of the episode

    size = core_env.size
    n_bins = 50
    grid_count = np.zeros((n_bins, n_bins), dtype=int)
    grid_reward = np.zeros((n_bins, n_bins), dtype=np.float64)

    for eps in tqdm(range(1,n_test+1), ncols = 100, colour = "#33FF00", desc = f"Evaluating {result_num}..."):
        # if eps == 0:
        #     if DEFAULT_CAMERA=="overhead_camera": pyautogui.press('tab')
        #     if ENABLE_FRAME: pyautogui.press('e') 
        #     if not RENDER_EVERY_FRAME: pyautogui.press('d')
        done = False
        acc_rew = 0
        step_count = 0

        while not done:
            action, _ = model_load.predict(obs, deterministic=True)
            # print(f"{action}           ", end='\r')
            nobs, rew, term, trunc, info = test_env.step(action)
            acc_rew += rew
            step_count += 1
            # if render_mode == "human":  # visual
            #     print(f"action: {action} | rew_appr: {info.get('rew_approach',-10.0):10.6f}                      ", end="\r")
            done = term or trunc

            if rew == rew_goal:
                success_count += 1

            if not done:
                obs = nobs
            else: 
                obs, info = test_env.reset()
                rew_hist_normalized[eps-1] = acc_rew/step_count
                agent_init_list.append(info["agent_init"])
                rew_head_list.append(info["rew_head"])
                

        if eps % plot_freq == 0:
            for idx, agent_init in enumerate(agent_init_list):
                x, y, theta = agent_init
                
                ix = int((x+size)/(2*size) * (n_bins))
                iy = int((y+size)/(2*size) * (n_bins))

                if ix >= 100 or iy >= 100:
                    print(x, y, ix, iy)
                grid_count[ix, iy] += 1
                count = grid_count[ix, iy]
                alpha = (count-1)/count
                grid_reward[ix, iy] = alpha * grid_reward[ix, iy] + (1-alpha) * rew_head_list[idx]
                
            tick_labels = np.round(np.linspace(-size, size ,n_bins),1)
            fig, axes = plt.subplots(1,2, figsize=(20,8))

            # --- plot of the agent's spawn frequency in different sub-divisions of the environment
            axes[0] = sns.heatmap(grid_count.T, ax=axes[0], cmap = 'plasma', 
                                xticklabels=tick_labels,yticklabels=tick_labels)
            axes[0].invert_yaxis()
            axes[0].set_aspect('equal')
            axes[0].set_title(f'Agent spawn frequency in {result_num} | eps {eps:05d}')
            axes[0].set_xlabel('x')
            axes[0].set_ylabel('y')

            # --- plot of the agent's final heading reward when starting in different sub-divisions of the environment
            axes[1] = sns.heatmap(grid_reward.T, ax=axes[1], cmap = 'plasma',
                                xticklabels=tick_labels, yticklabels=tick_labels)
            axes[1].invert_yaxis()
            axes[1].set_aspect('equal')
            axes[1].set_title(f'Average reward in {result_num} | eps {eps:05d}')

            plt.savefig(os.path.join(result_path,f"spawn_reward_{eps:05}.png"))
            # plt.close()

        # Report the success rate throughout the most recent 100 runs
        if eps % plot_freq == 0:
            print(f"\r\nSuccess rate from run {eps-plot_freq+1} to {eps} is {success_count/plot_freq*100:.2f}%             ")
            success_count = 0

    test_env.close()

    # print(agent_init_list)
    
    # print(rew_hist_normalized)
    # histogram of the normalized result
    plt.figure()
    plt.hist(rew_hist_normalized, bins=50)
    plt.title("Distribution of normalized rewards")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    plt.savefig(os.path.join(result_path,f"normalized_reward.png"))
    plt.close()