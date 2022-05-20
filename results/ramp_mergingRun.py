import os
import time

import gym

import environments
from agents.policy_gradient_agents.PPO import PPO
from agents.actor_critic_agents.DDPG import DDPG
from agents.actor_critic_agents.SAC import SAC
from agents.actor_critic_agents.TD3 import TD3
from agents.actor_critic_agents.A3C import A3C
from agents.Trainer import Trainer
from utilities.data_structures.Config import Config

env_title="RampMerging-v1"

config = Config()
config.env_title = env_title
config.seed = 1
config.environment = gym.make(env_title)
config.num_episodes_to_run = 6000
config.file_to_save_data_results = "data_and_graphs/RampMergeSafe_SAC.pkl"
config.file_to_save_results_graph = "data_and_graphs/RampMergSafe_SAC.png"
config.show_solution_score = False
config.visualise_individual_results = True
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 1
config.use_GPU = True
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.save_model = True
config.log_base = time.strftime("%Y%m%d%H%M%S", time.localtime())
config.save_model_freq = 100

config.retrain = False
config.resume = False
config.resume_path = 'E:\SUMO_PY\SUMO_RampMerge_1\\results\Models\RampMerging-v1\TD3\\20210426103716\TD3_6000_997.model'

# data_results_root = os.path.join(os.path.dirname(__file__)+"/data_and_graphs/RampMerging", config.log_base)
# while os.path.exists(data_results_root):
#     data_results_root += '_'
# os.makedirs(data_results_root)
# config.file_to_save_data_results = os.path.join(data_results_root, "data.pkl")
# config.file_to_save_results_graph = os.path.join(data_results_root, "data.png")

config.hyperparameters = {
    "Policy_Gradient_Agents": {
            "learning_rate": 1e-4,
            "linear_hidden_units": [64, 64],
            "final_layer_activation": "TANH",
            "learning_iterations_per_round": 10,
            "discount_rate": 0.9,
            "batch_norm": False,
            "clip_epsilon": 0.2,
            "episodes_per_learning_round": 3,
            "normalise_rewards": True,
            "gradient_clipping_norm": 5,
            "mu": 0.0,
            "theta": 0.15,
            "sigma": 0.2,
            "epsilon_decay_rate_denominator": 1,
            "clip_rewards": False
        },

    "Actor_Critic_Agents": {
            "Actor": {
                "learning_rate": 3e-4,
                "linear_hidden_units": [256]*2,
                "final_layer_activation": None,
                "batch_norm": False,
                "tau": 0.001,  # 0.005
                "gradient_clipping_norm": 3,
                "initialiser": "Xavier"
            },

            "Critic": {
                "learning_rate": 3e-4,
                "linear_hidden_units": [256]*2,
                "final_layer_activation": None,
                "batch_norm": False,
                "buffer_size": 1000000,
                "tau": 0.001,
                "gradient_clipping_norm": 3,
                "initialiser": "Xavier"
            },

        "min_steps_before_learning": 1000, #for SAC only
        "batch_size": 256,
        "discount_rate": 0.99,
        "mu": 0.0,  # for O-H noise
        "theta": 0.15,  # for O-H noise
        "sigma": 0.25,  # for O-H noise
        "action_noise_std": 0.2,  # for TD3
        "action_noise_clipping_range": 0.5,  # for TD3
        "update_every_n_steps": 1024,
        "learning_updates_per_learning_session": 1024,
        "automatically_tune_entropy_hyperparameter": True,
        "entropy_term_weight": None,
        "add_extra_noise": True,
        "do_evaluation_iterations": False,
        "clip_rewards": False

    }

}

if __name__ == "__main__":
    AGENTS = [DDPG]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()

# SAC, ,




