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

    default_hyperparam_dict = {
        "lr":               1e-3,
        "buffer_size":      2e6,
        "learn_start":      1e5,
        "batch_size":       256,
        "tau":              3e-3,
        "gamma":            0.99,
        "train_freq":       2,
        "grad_step":        4,
        "act_noise_std":    0.05,
        "n_steps":          1,
        "policy_delay":     4,
        "tpolicy_noise":    0.05,
        "tpolicy_clip":     0.25,
        "model_dir":        "Nav2D_TD3_SB3_results",
        "tensorboard_dir":  "Nav2D_TD3_SB3_tensorboard"}

    default_reward_scale = {
        "rew_dist_scale":           0.5,
        "rew_dist_approach_scale":  25.0,
        "rew_head_scale":           1.0,
        "rew_head_approach_scale":  25.0,
        "rew_time":                 -0.01,
        "rew_goal_scale":           5_000.0,
        "rew_obst_scale":           -1_000.0}

    default_randomization_options = {"agent_freq": 1, "goal_freq": 1}

    def train(num_runs: int = 500,
              steps_per_run: int = 20_000,
              models_to_save: int = 25,
              hyperparam_dict: dict = default_hyperparam_dict,
              reward_scale: dict = default_reward_scale,
              randomization_options: dict = default_randomization_options
              ):

        # Environment vectorization
        n_proc = 24

        print("Making subprocess vectorized environments!")
        env = make_vec_env("Nav2D-v0", 
                            n_envs=n_proc, 
                            seed=73,
                            env_kwargs={"max_episode_steps": 1_500,
                                        "reward_scale_options": reward_scale,
                                        "randomization_options": randomization_options
                                        },
                            vec_env_cls=SubprocVecEnv, 
                            vec_env_kwargs=dict(start_method='forkserver'))
    
        n_actions = env.get_attr("action_space")[0].shape[0]        # obtain the size of the action_space from the list of action_space among the n_proc subprocess envs

        # Hyperparameters
        learning_rate   = float(hyperparam_dict.get("lr", 1e-3))
        buffer_size     = int(hyperparam_dict.get("buffer_size",2e6))
        learning_starts = int(hyperparam_dict.get("learn_start", 1e5))
        batch_size      = int(hyperparam_dict.get("batch_size",256))
        tau             = float(hyperparam_dict.get("tau", 3e-3))
        gamma           = float(hyperparam_dict.get("gamma", 0.99))
        train_freq      = int(hyperparam_dict.get("train_freq", 2))
        gradient_steps  = int(hyperparam_dict.get("grad_step", 4))
        act_noise_std   = int(hyperparam_dict.get("act_noise_std", 0.05))
        action_noise    = NormalActionNoise(mean=np.zeros(n_actions), sigma=act_noise_std*np.ones(n_actions))     
        n_steps         = int(hyperparam_dict.get("n_steps", 1))
        policy_delay    = int(hyperparam_dict.get("policy_delay", 4))
        target_policy_noise = float(hyperparam_dict.get("tpolicy_noise", 0.05))
        target_noise_clip   = float(hyperparam_dict.get("tpolicy_clip", 0.25))
        model_dir           = hyperparam_dict.get("model_dir", "Nav2D_TD3_SB3_results")
        tensorboard_dir     = hyperparam_dict.get("tensorboard_dir", "Nav2D_TD3_SB3_tensorboard")
        verbose=0
        dir_path = os.path.dirname(os.path.abspath(__file__))
        tensorboard_log_dir=os.path.join(dir_path,"results",tensorboard_dir)

        pi_arch = [512, 256]
        qf_arch = [512, 256]
        policy_kwargs=dict(activation_fn=torch.nn.ReLU,
                        net_arch=dict(pi=pi_arch, qf=qf_arch))
        use_custom_policy = False
        cuda_enabled = True
        
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
                tensorboard_log=tensorboard_log_dir)
        
        # Training code
        # run parameters:
        number_of_runs = num_runs
        steps_per_run = steps_per_run
        models_to_save = models_to_save
        model_save_freq = int(number_of_runs / models_to_save)

        # model saving parameters:
        base_path = os.path.join(dir_path, "results", model_dir)
        result_number = f"result_{len(os.listdir(base_path))-1:05d}"
        results_path = os.path.join(base_path, result_number)

        curriculum_callback = CurriculumCallback(success_window=50, threshold=0.75)

        # Save the result-params mapping into a json file
        trial_to_param_path = os.path.join(base_path,'trial_to_param.json')
        if os.path.exists(trial_to_param_path):
            with open(trial_to_param_path, "r") as f:
                data = json.load(f)
        else:
            data = {result_number: ""}

        hyperparam_codified = f"{learning_rate}_{buffer_size}_{learning_starts}_{batch_size}_{tau}_{gamma}_"
        hyperparam_codified += f"{train_freq}_{gradient_steps}_{act_noise_std}_{n_steps}_{policy_delay}_{target_policy_noise}_{target_noise_clip}_"
        hyperparam_codified += f"{reward_scale['rew_head_scale']}_{reward_scale['rew_head_approach_scale']}_{reward_scale['rew_dist_scale']}_{reward_scale['rew_dist_approach_scale']}_{reward_scale['rew_goal_scale']}_{reward_scale['rew_obst_scale']}_"
        hyperparam_codified += f"{randomization_options['agent_freq']}_{randomization_options['goal_freq']}"

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
                vec_norm_env = model.get_env()
                vec_norm_env.save(os.path.join(results_path, f"norm_stats_{run}.pkl"))

        # save the last model
        model_path = os.path.join(results_path, f"run_{run}")
        norm_path = os.path.join(results_path, f"norm_stats_{run}.pkl")
        vec_norm_env = model.get_env()

        model.save(model_path)
        vec_norm_env.save(norm_path)

        # close environment when done:
        env.close()

        return model_path, norm_path
    
    def eval_policy(env: gym.Env, 
         n_evals: int, 
         model):
        
        # unwrap the environment to obtain maximum episode length and arena size
        if isinstance(env, DummyVecEnv):
            max_eps_length = env.envs[0].spec.max_episode_steps
            size = env.envs[0].unwrapped.size
            is_vec_env = True
        elif isinstance(env, (SubprocVecEnv, VecNormalize)):
            max_eps_length = env.get_attr("spec")[0].max_episode_steps
            size = env.get_attr("size")[0]
            is_vec_env = True
        else:    
            max_eps_length = env.spec.max_episode_steps
            size = env.unwrapped.size
            is_vec_env = False

        max_env_size = np.sqrt(2* (2*size)**2)

        # empty list to later store the results
        success_window  = deque(maxlen=n_evals)
        eps_len_window  = deque(maxlen=n_evals)
        dfinal_window   = deque(maxlen=n_evals)
        
        if is_vec_env:      # evaluating with a SB3 vectorized environment
            n_envs = env.num_envs
            obss = env.reset()
            eps_evaluated = 0

            while eps_evaluated <= n_evals:
                action, _ = model.predict(obss, deterministic = True)
                original_obss = env.get_original_obs()
                nobss, rews, dones, infos = env.step(action)

                for i, (done, info) in enumerate(zip(dones, infos)):
                    if done:
                        eps_evaluated += 1
                        success_window.append(info.get("is_success", False))
                        eps_len_window.append(info.get('episode', {}).get('l', None))
                        dfinal_window.append(original_obss[i][2]) 
                        # TODO - how to grab the unnormalized final distance from the environment that terminated
                obss = nobss
        else:               # TODO - evaluation for a core gymnasium environment
            pass

        mean_success    = np.mean(success_window)
        mean_ep_len     = np.mean(eps_len_window)/max_eps_length
        mean_dfinal     = np.mean(dfinal_window)/max_env_size

        return mean_success, mean_ep_len, mean_dfinal

    class CurriculumCallback(BaseCallback):
        def __init__(self, 
                     success_window:int = 100, 
                     threshold:float=0.9, 
                     debug:bool=False):
            super().__init__()
            self.success_threshold = threshold
            self.success_window = success_window
            self.buffer = deque(maxlen=success_window)

            self.n_levels = 40
            self.goal_bound_levels = np.linspace(0,1,self.n_levels+1)
            self.goal_level_idx = 0
            self.agent_bound_levels = np.linspace(0,1,self.n_levels+1)
            self.agent_level_idx = 0

            self.explore_boost_steps = 0     # keep count of the number of steps with increased exploration (higher action noise)
            self.explore_boost_active = False

            self.cooldown_eps = int(2000/self.n_levels)
            self.cooldown_active = True
            self.cooldown_remaining = 100

            self.term_cond = {
                "is_success": 0,
                "obstacle_cond": 0,
                "dist_progress_cond": 0,
                "head_progress_cond": 0,
                "TimeLimit.truncated": 0
            }

            self.debug=debug

        def _on_step(self) -> bool:
            # grab info dict from each parallel env
            infos = self.locals["infos"]
            dones = self.locals["dones"]    
            
            for i, (done, info) in enumerate(zip(dones, infos)):
                if done:
                    self.buffer.append(info.get("is_success", False))       # if truncated, there's no is_success => False. If not truncated, then either True or False
                    if self.cooldown_active: self.cooldown_remaining -= 1
                    for k in self.term_cond.keys():
                        self.term_cond[k] += int(info.get(k, False))

            # cooldown episode countdown after curriculum advancement
            if self.cooldown_active and (self.cooldown_remaining <= 0): 
                if self.debug: print("\nCooldown finished")
                self.cooldown_active = False
                self.cooldown_remaining = 0

            # curriculum advancement logic
            if len(self.buffer) >= self.success_window:

                success_rate = sum(self.buffer)/len(self.buffer)
                print(f"Current training success {success_rate:3.2f}", end="\r")

                # curriculum advancement
                if (not self.explore_boost_active
                    and not self.cooldown_active
                    and success_rate >= self.success_threshold):
                    # print the breakdown of termination conditions before curriculum advancement
                    self._report_termination_cond()         

                    # only advance agent curriculum until agent curriculum reaches 50% progress
                    if self.agent_level_idx < (len(self.agent_bound_levels) - 1) // 2:      
                        self._advance_agent_curriculum()
                    else:
                    # alternate between agent and goal randomization afterward
                        if (self.agent_level_idx + self.goal_level_idx) % 2 != 0 and (self.agent_level_idx < self.n_levels):           
                            self._advance_agent_curriculum()
                        elif self.goal_level_idx < self.n_levels:
                            self._advance_goal_curriculum()

                    if self.agent_level_idx == self.n_levels and self.goal_level_idx == self.n_levels:
                        self.success_threshold += 0.1
                    self.buffer.clear()
                    self._start_cooldown()
                    self.term_cond = dict.fromkeys(self.term_cond, 0)       # reset all termination condition counts to 0

            # countdown of episodes with boosted exploration
            if self.explore_boost_steps > 0:
                self.explore_boost_steps -= 1
                if self.explore_boost_steps == 0:
                    self.explore_boost_active = False
                    vec_noise = self.model.action_noise
                    for i in vec_noise.noises:
                        i._sigma = self.noise_std_backup.copy()
                    
                    if self.debug: print(f"\nFinished with steps of increased exploration, noise reset to {i._sigma} \n")
            return True     # to continue training, return True, else return False

        def _start_cooldown(self):
            '''Function to count down the episodes of no curriculum advancing right after an advancement'''
            self.cooldown_active = True
            self.cooldown_remaining = self.cooldown_eps
            if self.debug: print(f"Starting cooldown for {self.cooldown_eps} episodes\n")

        def _boost_explore(self, num_steps):
            self.explore_boost_steps = num_steps
            self.explore_boost_active = True
            vec_noise = self.model.action_noise             # VectorizedActionNoise, which is a vector of the action noise (NormalActionNoise)
            noise = vec_noise.noises                        # list of NormalActionNoise instance of one environment
            self.noise_std_backup = noise[0]._sigma.copy()  # keep a copy of the original noise stdev

            for i in noise:
                i._sigma = [x+0.05 for x in i._sigma]     # increase action noise stdev for all child envs
            if self.debug: print(f"Action noise => {i._sigma} for {self.explore_boost_steps} steps")

        def _report_termination_cond(self):
            success_count  = self.term_cond.get('is_success',            'no is_success')
            obstacle_count = self.term_cond.get('obstacle_cond',         'no obstacle_cond')
            stall_d_count  = self.term_cond.get('dist_progress_cond',    'no dist_progress_cond')
            stall_h_count  = self.term_cond.get('head_progress_cond',    'no head_progress_cond')
            trunc_count    = self.term_cond.get('TimeLimit.truncated',   'no TimeLimit.truncated')
            eps_sum = success_count + obstacle_count + stall_d_count + stall_h_count + trunc_count

            print(f"\nA={self.agent_bound_levels[self.agent_level_idx]*100:5.2f}% | G={self.goal_bound_levels[self.goal_level_idx]*100:5.2f}% | "
                    f"success={success_count} ({success_count/eps_sum*100:5.2f}%) | "
                    f"obstacle={obstacle_count} ({obstacle_count/eps_sum*100:5.2f}%) | "
                    f"stall_d={stall_d_count} ({stall_d_count/eps_sum*100:5.2f}%) | "
                    f"stall_h={stall_h_count} ({stall_h_count/eps_sum*100:5.2f}%) | "
                    f"trunc={trunc_count} ({trunc_count/eps_sum*100:5.2f}%)")

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
            self.current_agent_bound = vec.get_attr("agent_bound")[0]      # get the actual goal_bound from one environment
            if self.debug: print(f"Increasing agent bound - level {self.agent_level_idx:2d} | agent_bound = {self.current_agent_bound:4.3f} | success_window = {self.success_window}")

            #--- boost exploration for several steps
            self._boost_explore(num_steps=500)

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
            self.current_goal_bound = vec.get_attr("goal_bound")[0]      # get the actual goal_bound from one environment
            if self.debug: print(f"Increasing goal bound - level {self.goal_level_idx:2d} | goal_bound = {self.current_goal_bound:4.3f} | success_window = {self.success_window}")

            #--- boost exploration for several steps
            self._boost_explore(num_steps=500)

    def objective_rew_scale(trial):        
        reward_scale = {
            "rew_dist_scale":           trial.suggest_categorical("w_dist", choices=[0.1, 0.25, 0.5, 1.0]),
            "rew_dist_approach_scale":  trial.suggest_categorical("w_dist_app", choices=[10.0, 25.0, 50.0]),
            "rew_head_scale":           trial.suggest_categorical("w_head", choices=[0.1, 0.25, 0.5, 1.0]),
            "rew_head_approach_scale":  trial.suggest_categorical("w_head_app", choices=[10.0, 25.0, 50.0]),
            "rew_time":                 trial.suggest_categorical("w_time", choices=[-0.01, -0.05, -0.1, -0.25]),
            "rew_goal_scale":           5_000,
            "rew_obst_scale":           -1_000
        }

        hyperparam_dict = {
            "model_dir":        "Nav2D_TD3_SB3_optuna_results",
            "tensorboard_dir":  "Nav2D_TD3_SB3_optuna_tensorboard"
        }

        model_path, norm_path = train(num_runs=250,
                                      steps_per_run=20_000,
                                      models_to_save=5,
                                      hyperparam_dict=hyperparam_dict,
                                      reward_scale=reward_scale)
        
        # Evaluate the policy
        print("Evaluating......")
        n_evals = 500
        eval_env = make_vec_env("Nav2D-v0", 
                                n_envs=4,
                                env_kwargs={"max_episode_steps": 1_000,
                                            "is_eval": True,
                                            "randomization_options": default_randomization_options,
                                            "reward_scale_options": default_reward_scale
                                            },
                                vec_env_cls=SubprocVecEnv,
                                vec_env_kwargs=dict(start_method='forkserver'))
        eval_env = VecNormalize.load(norm_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
        model = TD3.load(model_path, env=eval_env)
        success, ep_len, dfinal = eval_policy(env=eval_env, n_evals=n_evals, model=model)
        eval_env.close()

        return round(success,3), round(ep_len,3), round(dfinal,3)
    
    def rew_scale_optuna():
        study_name = "rew_scale_dec11"
        study = optuna.create_study(storage=f"sqlite:///python/environments/results/optuna_study.db", 
                                    study_name=study_name, 
                                    load_if_exists=True,
                                    direction='maximize')
        study.optimize(objective_rew_scale, n_trials=50)

    # # RUN OPTUNA STUDY
    rew_scale_optuna()
    

    # # RUN ONE TRAINING
    # for i in range(1):
    #     train(num_runs=100,
    #           steps_per_run=50_000, 
    #           models_to_save=25,
    #           hyperparam_dict=default_hyperparam_dict,
    #           reward_scale=default_reward_scale,
    #           randomization_options=default_randomization_options)

if __name__=="__main__":
    main()