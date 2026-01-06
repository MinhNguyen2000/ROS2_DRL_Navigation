# ===================================================================
# This script is used for training SAC using stable baselines 3 with 
# SubprocVecEnv for multi-core processing, with either the "forkserver" 
# or the "spawn" method

# These methods only work when the training code is wrapped in a 
# if __name__ == "main" block
# ===================================================================
# imports:
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
import torch

import gymnasium as gym
import nav2d
import numpy as np
import os, json
from datetime import datetime
from tqdm import tqdm
import optuna

# Ignore User Warnings (for creating a new folder to save policies):
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# study_name = "model_params_dec13"
# storage = "sqlite:///python/environments/results/optuna_results.db"
# optuna.delete_study(study_name = study_name, storage = storage)

# define initialization function:
def init_model(hyperparameters : dict, 
               reward_scale : dict = {str, float},
               randomization_options : dict = {str, int},
               normalize : bool = False):
    # environment vectorization settings:
    n_proc = 24

    # max episode steps:
    max_episode_steps = 2500

    # make the vectorized environment:
    env = make_vec_env("Nav2D-v0",
                        n_envs = n_proc,
                        env_kwargs = {"max_episode_steps" : max_episode_steps,
                                     "reward_scale_options" : reward_scale,
                                     "randomization_options" : randomization_options},
                        vec_env_cls = SubprocVecEnv,
                        vec_env_kwargs = dict(start_method = "forkserver"),
                        seed = 42)
    
    # wrap environments in a normalization wrapper:
    if normalize:
        env = VecNormalize(env, norm_obs = True, norm_reward = True)

    # make the model:
    model = SAC(policy = hyperparameters["policy"],
                env = env,
                buffer_size = hyperparameters["buffer_size"],
                batch_size = hyperparameters["batch_size"],
                tau = hyperparameters["tau"],
                gamma = hyperparameters["gamma"],
                ent_coef = hyperparameters["ent_coef"],
                train_freq = hyperparameters["train_freq"],
                learning_starts = hyperparameters["learning_starts"],
                target_update_interval = hyperparameters["target_update_interval"],
                gradient_steps = hyperparameters["gradient_steps"],
                target_entropy = hyperparameters["target_entropy"],
                action_noise = hyperparameters["action_noise"],
                verbose = hyperparameters["verbose"],
                device = "cuda" if hyperparameters["gpu"] else "cpu",
                policy_kwargs = hyperparameters["policy_kwargs"],
                tensorboard_log = hyperparameters["tensorboard_log"])

    # apply custom learning rates:
    model.actor.optimizer = torch.optim.Adam(model.actor.parameters(), lr = hyperparameters["actor_lr"])
    model.critic.optimizer = torch.optim.Adam(model.critic.parameters(), lr = hyperparameters["critic_lr"])

    # return to user:
    return env, model

# evaluation function for optuna studies:
def eval_policy(eval_env: gym.Env, 
                num_evals: int,
                normalize: bool,
                model):
    
    # objective params:
    returns = []
    successes = []

    # for the number of episodes to eval for:
    for _ in range(num_evals):
        # reset:
        if normalize:
            obs = eval_env.reset()
        else:
            obs, _ = eval_env.reset()

        # initialize:
        episode_return = 0
        done = False

        # while False:
        while not done:
            # get action from policy:
            action, _ = model.predict(obs, deterministic = True)

            # step environment:
            if normalize:
                nobs, reward, term, info = eval_env.step(action)
                episode_return += reward
                done = term
            else:
                nobs, reward, term, trunc, _  = eval_env.step(action)
                episode_return += reward
                done = term or trunc

            # advance observation, reset if not:
            if not done:
                obs = nobs
            else:
                if normalize:
                    successes.append(info[0].get("is_success", False))
                else:
                    obs, _ = eval_env.reset()

        # advance episodic return:
        returns.append(episode_return)

    return np.mean(returns), np.mean(successes)
    # # reset the environment:
    # if normalize:
    #     obs = env.reset()
    # else:
    #     obs, _ = env.reset()

    # # for each episode in the num_evals:
    # for _ in range(num_evals):

    #     # initialize:
    #     eval_rew = 0
    #     done = False

    #     # while not done:
    #     while not done:
    #         action, _ = model.predict(obs, deterministic = True)
    #         if normalize:
    #             nobs, reward, term, info = env.step(action)
    #             eval_rew += reward
    #             # print(f"abs_diff: {nobs[0][0:2]} | reward: {reward}")
    #             done = term
    #         else:
    #             nobs, reward, term, trunc, _ = env.step(action)
    #             done = term or trunc

    #         # advance observation, reset if not:
    #         if not done:
    #             obs = nobs
    #         else:
    #             if normalize:
    #                 successes.append(info[0].get("is_success", False))
    #                 # print(info, end = '\r')
    #             else: 
    #                 obs, _ = env.reset()

    #     eval_rew_hist.append(eval_rew)

    # # return when done:
    # return np.mean(eval_rew_hist).round(3), np.mean(successes)

# function for training a given model:
def train_model(model,
                env,
                dir_path,
                reward_scale : dict,
                randomization_options : dict,
                hyperparameters : dict,
                number_of_runs : int = 100,
                steps_per_run : int = 25000,
                normalize : bool = False):
    # set the model saving frequency:
    model_save_freq = max(int(number_of_runs / 20), 1)

    # set the model saving path:
    base_path = os.path.join(dir_path, "results", "Nav2D_SAC_SB3_results")
    result_number = f"result_{len(os.listdir(base_path)):03d}"
    results_path = os.path.join(base_path, result_number)

    # save the result-params mapping into a json file:
    trial_to_param_path = os.path.join(base_path,'trial_to_param.json')
    if os.path.exists(trial_to_param_path):
        with open(trial_to_param_path, "r") as f:   
            data = json.load(f)
    else:
        data = {result_number: ""}

    hyperparam_codified = f"{hyperparameters["policy_kwargs"]}_"
    hyperparam_codified += f"{hyperparameters["actor_lr"]}_{hyperparameters["critic_lr"]}_{hyperparameters["buffer_size"]}_{hyperparameters["batch_size"]}_{hyperparameters["tau"]}_{hyperparameters["gamma"]}_"
    hyperparam_codified += f"{hyperparameters["train_freq"]}_{hyperparameters["gradient_steps"]}_{hyperparameters["ent_coef"]}_{hyperparameters["target_update_interval"]}_{hyperparameters["target_entropy"]}_"
    hyperparam_codified += f"{reward_scale['rew_head_scale']}_{reward_scale["rew_head_approach_scale"]}_{reward_scale['rew_dist_scale']}_{reward_scale['rew_dist_approach_scale']}_{reward_scale['rew_goal_scale']}_{reward_scale['rew_obst_scale']}_"
    # hyperparam_codified += f"{reward_scale['rew_head_scale']}_{reward_scale['rew_dist_scale']}_{reward_scale['rew_goal_scale']}_{reward_scale['rew_obst_scale']}_"
    hyperparam_codified += f"{randomization_options['agent_freq']}_{randomization_options["goal_freq"]}_{randomization_options["obstacle_freq"]}"

    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    hyperparam_codified_time = f"{timestamp}_" + hyperparam_codified

    data[result_number] = hyperparam_codified_time

    with open(trial_to_param_path, "w") as f:
        json.dump(data, f, indent=2)
        
    # train using model.learn():
    for run in tqdm(range(1, number_of_runs + 1), ncols = 100, colour = "#33FF00", desc = f"{result_number} training progress"):
        # learn every run:
        model.learn(total_timesteps = steps_per_run, tb_log_name = f"{result_number}", reset_num_timesteps = False)

        # set the run saving path:
        run_dir = os.path.join(results_path, f"run_{run}")

        # save a model + stats every now and then:
        if run % model_save_freq == 0:
            # save things:
            model.save(os.path.join(run_dir, f"run_{run}"))
            vec_norm_env = model.get_env()
            vec_norm_env.save(os.path.join(run_dir, "vec_norm_env_stats.pkl"))

    # save the last model:
    model.save(os.path.join(run_dir, f"run_{run}"))

    # save the last normalization stats, if normalize flag is True:
    if normalize:
        vec_norm_env = model.get_env()
        vec_norm_env.save(os.path.join(run_dir, "vec_norm_env_stats.pkl"))

    # close the environment:
    env.close()

    # return path to model, normalization stats:
    model_path = os.path.join(run_dir, f"run_{run}")
    norm_path = os.path.join(run_dir, "vec_norm_env_stats.pkl")

    return model_path, norm_path
    
# objective function for optuna:
def objective_model_params(trial):
    # set the tensorboard logging directory:
    dir_path = os.path.dirname(os.path.abspath(__file__))
    tensorboard_log_dir = os.path.join(dir_path, "results", "Nav2D_SAC_SB3_tensorboard")

    # set the training parameters:
    number_of_runs = 100
    steps_per_run = 50_000
    normalize = True

    # define the randomization options:
    randomization_options = {"agent_freq" : 1,
                             "goal_freq" : 1,
                             "obstacle_freq" : 1}

    # define the desired reward scale:
    reward_scale = {"rew_head_scale" : 1.0,
                    "rew_head_approach_scale" : 25.0,
                    "rew_dist_scale" : 0.5,
                    "rew_dist_approach_scale" : 25.0,
                    "rew_goal_scale" : 5000.0,
                    "rew_obst_scale" : -1000.0, 
                    "rew_time" : -0.01}
    
    # suggest values for hyperparameters:
    # gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.99, 0.995, 0.999])
    gamma = 0.99
    # actor_lr = trial.suggest_categorical("actor_lr", [1e-5, 2.5e-5, 5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3])
    # critic_lr = trial.suggest_categorical("critic_lr", [1e-5, 2.5e-5, 5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3])
    actor_lr = 3e-5
    critic_lr = 3e-4
    # buffer_size = trial.suggest_categorical("buffer_size", [int(1e6), int(1.5e6), int(2e6), int(2.5e6)])
    buffer_size = int(2e6)
    batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048])
    # tau = trial.suggest_float("tau", low = 1e-3, high = 1e-2, step = 1e-3)
    tau = 5e-3
    ent_coef = trial.suggest_categorical("ent_coeff", ["auto", "auto_0.1"])
    train_freq = trial.suggest_int("train_freq", low = 1, high = 4, step = 1)
    target_update_interval = trial.suggest_int("target_update_interval", low = 1, high = 4, step = 1)
    gradient_steps = trial.suggest_int("gradient_steps", low = 1, high = 4, step = 1)
    target_entropy = trial.suggest_categorical("target_entropy", ["auto", "-2"])
    
    # define the hyperparameters:
    actor_arch = [256, 256]
    critic_arch = [256, 256]
    policy_kwargs = {"net_arch" : {"pi" : actor_arch, "qf" : critic_arch}}

    hyperparameters = {"policy" : "MlpPolicy",
                       "gamma" : gamma,
                       "actor_lr" : actor_lr,
                       "critic_lr" : critic_lr,
                       "buffer_size" : buffer_size,
                       "batch_size" : batch_size,
                       "tau" : tau,
                       "ent_coef" : ent_coef,
                       "train_freq" : train_freq, 
                       "learning_starts" : 50_000,
                       "target_update_interval" : target_update_interval,
                       "gradient_steps" : gradient_steps, 
                       "target_entropy" : target_entropy, 
                       "action_noise" : None,
                       "verbose" : 0, 
                       "gpu" : True,
                       "policy_kwargs": policy_kwargs,
                       "tensorboard_log" : tensorboard_log_dir}
    
    # get the model and the environment:
    env, model = init_model(hyperparameters = hyperparameters, 
                            reward_scale = reward_scale, 
                            randomization_options = randomization_options,
                            normalize = normalize)
    
    # train the model:
    model_path, norm_path = train_model(model = model,
                                        env = env,
                                        dir_path = dir_path,
                                        reward_scale = reward_scale,
                                        randomization_options = randomization_options,
                                        hyperparameters = hyperparameters,
                                        number_of_runs = number_of_runs,
                                        steps_per_run = steps_per_run)

    # close old env:
    model.env.close()
    
    # need to now evaluate the policy:
    n_evals = 500

    # make a single environment:
    env = gym.make("Nav2D-v0",
            render_mode = "rgb_array",
            width = 1280, 
            height = 1280,
            default_camera_config = {"azimuth" : 90.0, "elevation" : -90.0, "distance" : 3, "lookat" : [0.0, 0.0, 0.0]}, 
            camera_id = 2, 
            max_episode_steps = 1000, 
            is_eval = False,
            reward_scale_options = reward_scale,
            randomization_options = randomization_options,
            visual_options = {8 : False})
    
    eval_env = DummyVecEnv([lambda: env])
    eval_env = VecNormalize.load(norm_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False
    model = SAC.load(model_path, env = eval_env)

    mean_return, mean_successes = eval_policy(eval_env = eval_env, 
                                              num_evals = n_evals, 
                                              model = model,
                                              normalize = normalize)
    eval_env.close()

    # return reward for optimizing:
    return mean_return, mean_successes

# main function:
def main(do_studies : bool = False,
         normalize : bool = False):
    # set the tensorboard logging directory:
    dir_path = os.path.dirname(os.path.abspath(__file__))
    tensorboard_log_dir = os.path.join(dir_path, "results", "Nav2D_SAC_SB3_tensorboard")

    # set the training parameters:
    number_of_runs = 100
    steps_per_run = 100_000

    # if not using optuna:
    if not do_studies:
        # reward_scale:
        for _ in range(5):
            reward_scale = {
                            "rew_dist_scale" : 2.0,               
                            "rew_dist_approach_scale" : 25.0,
                            "rew_head_scale" : 1.0,             
                            "rew_head_approach_scale" : 25.0,   
                            "rew_goal_scale" : 7_500,          
                            "rew_obst_scale" : -2_000,        
                            "rew_time" : -0.1}                 
            
            randomization_options = {"agent_freq" : 1,
                            "goal_freq" : 1,
                            "obstacle_freq" : 1}
            
            # model hyperparameters:
            actor_arch = [256, 256]
            critic_arch = [256, 256]
            policy_kwargs = {"net_arch" : {"pi" : actor_arch, "qf" : critic_arch}}

            hyperparameters = {"policy" : "MlpPolicy",
                            "actor_lr" : 3e-5,
                            "critic_lr" : 3e-4,
                            "buffer_size" : int(2.5e6),
                            "learning_starts" : int(1e5),
                            "batch_size" : 256,
                            "tau" : 5e-3,
                            "gamma" : 0.99,
                            "train_freq" : 2,
                            "target_update_interval" : 1,
                            "gradient_steps" : 4,
                            "ent_coef" : "auto",
                            "target_entropy" : "auto",
                            "action_noise" : None,
                            "verbose" : 0, 
                            "gpu" : True,
                            "policy_kwargs": policy_kwargs,
                            "tensorboard_log" : tensorboard_log_dir}
            
            # get the model and the environment:
            env, model = init_model(hyperparameters = hyperparameters, 
                                    reward_scale = reward_scale,
                                    randomization_options = randomization_options,
                                    normalize = normalize)
            
            # train the model:
            _, _ = train_model(model = model, 
                        env = env,
                        dir_path = dir_path,
                        reward_scale = reward_scale,
                        randomization_options = randomization_options,
                        hyperparameters = hyperparameters,
                        number_of_runs = number_of_runs,
                        steps_per_run = steps_per_run,
                        normalize = normalize)
            
    # if using optuna:
    else:
        # set the study parameters:
        study_name = "model_params_dec13"
        directions = ["maximize", "maximize"]
        storage = "sqlite:///python/environments/results/optuna_results.db"
        load_if_exists = True

        # optuna.delete_study(study_name = study_name, storage = storage)

        # create the study object:
        study = optuna.create_study(study_name = study_name,
                                    directions = directions,
                                    storage = storage,
                                    load_if_exists = load_if_exists)
        
        # optimize the objective function:
        study.optimize(objective_model_params, n_trials = 100)

if __name__=="__main__":
    main(do_studies = False, 
         normalize = True)
