# ===================================================================
# This script is used for training SAC using stable baselines 3 with 
# SubprocVecEnv for multi-core processing, with either the "forkserver" 
# or the "spawn" method

# These methods only work when the training code is wrapped in a 
# if __name__ == "main" block
# ===================================================================
# imports:
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
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

# define initialization function:
def init_model(hyperparameters : dict, 
               reward_scale : dict = {str, float},
               randomization_options : dict = {str, int},
               normalize : bool = False):
    # environment vectorization settings:
    n_proc = 24

    # max episode steps:
    max_episode_steps = 1000

    # make the vectorized environment:
    env = make_vec_env("Nav2D-v0",
                        n_envs = n_proc,
                        env_kwargs = {"max_episode_steps" : max_episode_steps,
                                     "reward_scale_options" : reward_scale,
                                     "randomization_options" : randomization_options},
                        vec_env_cls = SubprocVecEnv,
                        vec_env_kwargs = dict(start_method = "spawn"))
    
    # wrap environments in a normalization wrapper:
    if normalize:
        env = VecNormalize(env, norm_obs = True, norm_reward = True, clip_obs = 10.0)

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
                tensorboard_log = hyperparameters["tensorboard_log"])

    # apply custom learning rates:
    model.actor.optimizer = torch.optim.Adam(model.actor.parameters(), lr = hyperparameters["actor_lr"])
    model.critic.optimizer = torch.optim.Adam(model.critic.parameters(), lr = hyperparameters["critic_lr"])

    # return to user:
    return env, model

# evaluation function for optuna studies:
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

    # return when done:
    return np.mean(eval_rew_hist).round(3)

# function for training a given model:
def train_model(model,
                dir_path,
                reward_scale : dict,
                randomization_options : dict,
                hyperparameters : dict,
                number_of_runs : int = 100,
                steps_per_run : int = 25000,
                normalize : bool = False):
    # set the model saving frequency:
    model_save_freq = max(int(number_of_runs / 10), 1)

    # set the model saving path:
    base_path = os.path.join(dir_path, "results", "Nav2D_SAC_SB3_results")
    result_number = f"result_{len(os.listdir(base_path)):03d}"
    results_path = os.path.join(base_path, result_number)
        
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

    # save the result-params mapping into a json file:
    trial_to_param_path = os.path.join(base_path,'trial_to_param.json')
    if os.path.exists(trial_to_param_path):
        with open(trial_to_param_path, "r") as f:   
            data = json.load(f)
    else:
        data = {result_number: ""}

    hyperparam_codified = f"{hyperparameters["actor_lr"]}_{hyperparameters["critic_lr"]}_{hyperparameters["buffer_size"]}_{hyperparameters["batch_size"]}_{hyperparameters["tau"]}_{hyperparameters["gamma"]}_"
    hyperparam_codified += f"{hyperparameters["train_freq"]}_{hyperparameters["gradient_steps"]}_{hyperparameters["ent_coef"]}_{hyperparameters["target_update_interval"]}_{hyperparameters["target_entropy"]}_"
    hyperparam_codified += f"{reward_scale['rew_head_scale']}_{reward_scale["rew_head_approach_scale"]}_{reward_scale['rew_dist_scale']}_{reward_scale['rew_goal_scale']}_{reward_scale['rew_obst_scale']}"
    hyperparam_codified += f"{randomization_options['agent_freq']}_{randomization_options["goal_freq"]}_{randomization_options["obstacle_freq"]}"

    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    hyperparam_codified_time = f"{timestamp}_" + hyperparam_codified

    data[result_number] = hyperparam_codified_time

    with open(trial_to_param_path, "w") as f:
        json.dump(data, f, indent=2)

# objective function for optuna:
def objective_model_params(trial):
    # set the tensorboard logging directory:
    dir_path = os.path.dirname(os.path.abspath(__file__))
    tensorboard_log_dir = os.path.join(dir_path, "results", "Nav2D_SAC_SB3_tensorboard")

    # set the training parameters:
    number_of_runs = 100
    steps_per_run = 25000

    # define the desired reward scale:
    reward_scale = {"rew_head_scale" : 10.0,
                    "rew_head_approach_scale" : 220.0,
                    "rew_dist_scale" : 3.5,
                    "rew_dist_approach_scale" : 200.0,
                    "rew_goal_scale" : 5000.0,
                    "rew_obst_scale" : -1000.0, 
                    "rew_time" : -0.25}
    
    # suggest values for hyperparameters:
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.99, 0.995, 0.999])
    actor_lr = trial.suggest_categorical("actor_lr", [1e-5, 2.5e-5, 5e-5, 1e-4, 2.5e-4, 5e-4])
    critic_lr = trial.suggest_categorical("critic_lr", [1e-5, 2.5e-5, 5e-5, 1e-4, 2.5e-4, 5e-4])
    buffer_size = trial.suggest_int("buffer_size", low = int(1e6), high = int(5e6), step = int(2.5e5))
    tau = trial.suggest_float("tau", low = 1e-3, high = 1e-2, step = 1e-3)
    ent_coef = trial.suggest_categorical("ent_coeff", ["auto", "auto_0.1"])
    train_freq = trial.suggest_int("train_freq", low = 1, high = 8, step = 1)
    target_update_interval = trial.suggest_int("target_update_interval", low = 1, high = 8, step = 1)
    gradient_steps = trial.suggest_int("gradient_steps", low = 1, high = 4, step = 1)
    target_entropy = trial.suggest_categorical("target_entropy", ["auto", -2])
    
    # define the hyperparameters:
    hyperparameters = {"policy" : "MlpPolicy",
                       "gamma" : gamma,
                       "actor_lr" : actor_lr,
                       "critic_lr" : critic_lr,
                       "buffer_size" : buffer_size,
                       "batch_size" : 4096,
                       "tau" : tau,
                       "ent_coef" : ent_coef,
                       "train_freq" : train_freq, 
                       "learning_starts" : 0,
                       "target_update_interval" : target_update_interval,
                       "gradient_steps" : gradient_steps, 
                       "target_entropy" : target_entropy, 
                       "action_noise" : None,
                       "verbose" : 0, 
                       "gpu" : True,
                       "tensorboard_log" : tensorboard_log_dir}
    
    # get the model and the environment:
    _, model = init_model(hyperparameters = hyperparameters, 
                            reward_scale = reward_scale)
    
    # train the model:
    train_model(model = model, 
                dir_path = dir_path,
                reward_scale = reward_scale,
                hyperparameters = hyperparameters,
                number_of_runs = number_of_runs,
                steps_per_run = steps_per_run)
    
    # close env:
    model.env.close()
    
    # evaluate the policy:
    n_evals = 25
    eval_env = gym.make("Nav2D-v0", max_episode_steps = 1_000, render_mode = "rgb_array", is_eval = True)
    mean_eval_rew = eval_policy(env = eval_env, 
                                num_evals = n_evals, 
                                model = model)
    eval_env.close()

    # return reward for optimizing:
    return mean_eval_rew

# main function:
def main(do_studies : bool = False,
         normalize : bool = False):
    # set the tensorboard logging directory:
    dir_path = os.path.dirname(os.path.abspath(__file__))
    tensorboard_log_dir = os.path.join(dir_path, "results", "Nav2D_SAC_SB3_tensorboard")

    # set the training parameters:
    number_of_runs = 500
    steps_per_run = 50000

    # if not using optuna:
    if not do_studies:
        # reward_scale:
        reward_scale = {"rew_head_scale" : 10.0,
                        "rew_head_approach_scale" : 220.0,
                        "rew_dist_scale" : 3.5,
                        "rew_dist_approach_scale" : 200.0,
                        "rew_goal_scale" : 5000.0,
                        "rew_obst_scale" : -1000.0, 
                        "rew_time" : -0.25}
        
        randomization_options = {"agent_freq" : 1,
                         "goal_freq" : 25,
                         "obstacle_freq" : 1}
        
        # model hyperparameters:
        hyperparameters = {"policy" : "MlpPolicy",
                           "gamma" : 0.99,
                           "actor_lr" : 3e-4,
                           "critic_lr" : 3e-4,
                           "buffer_size" : int(1e6),
                           "batch_size" : 512,
                           "tau" : 5e-3,
                           "ent_coef" : "auto",
                           "train_freq" : 2,
                           "learning_starts" : 0,
                           "target_update_interval" : 1,
                           "gradient_steps" : 2,
                           "target_entropy" : "-2",
                           "action_noise" : None,
                           "verbose" : 0, 
                           "gpu" : True,
                           "tensorboard_log" : tensorboard_log_dir}
        
        # get the model and the environment:
        _, model = init_model(hyperparameters = hyperparameters, 
                                reward_scale = reward_scale,
                                randomization_options = randomization_options,
                                normalize = normalize)
        
        # train the model:
        train_model(model = model, 
                    dir_path = dir_path,
                    reward_scale = reward_scale,
                    randomization_options = randomization_options,
                    hyperparameters = hyperparameters,
                    number_of_runs = number_of_runs,
                    steps_per_run = steps_per_run,
                    normalize = normalize)
            
    # if using optuna:
    else:
        # # set the study parameters:
        study_name = "model_params_nov13"
        direction = "maximize"
        storage = "sqlite:///python/environments/results/optuna_results.db"
        load_if_exists = True

        # optuna.delete_study(study_name = study_name, storage = storage)

        # create the study object:
        study = optuna.create_study(study_name = study_name,
                                    direction = direction,
                                    storage = storage,
                                    load_if_exists = load_if_exists)
        
        # optimize the objective function:
        study.optimize(objective_model_params, n_trials = 100)

if __name__=="__main__":
    main(do_studies = False, 
         normalize = True)
