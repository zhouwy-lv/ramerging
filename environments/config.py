import inspect
import logging
import torch
import os



class Settings:

    # Task
    # TASK = "TRAIN_DQN"  # "ST" or "TRAIN_DQN" or "RESUME_DQN" or "EVALUATE_DQN" or "EVALUATE_COMBINED_DDPG", etc.
    # NUM_EPISODES = 2000
    # GYM_ENVIRONMENT = "sumo-jerk-continuous-v0"  # "sumo-jerk-continuous-v0" or "sumo-jerk-v0", probably

    # Logging
    # LOG_DIR = "last_run"
    # FULL_LOG_DIR = "runs"
    # LOG_FILE = "out.log"
    # LOG_LEVEL = logging.INFO
    # MODEL_NAME = "runs/ddpg_simple_traffic_vary_start_extended"
    # INIT_MODEL_NAME = ""




    # Simulation
    TICK_LENGTH = 0.1
    MAX_POSITIVE_ACCELERATION = 4.5
    MAX_NEGATIVE_ACCELERATION = -6.0
    MINIMUM_NEGATIVE_JERK = -5.0
    MAXIMUM_POSITIVE_JERK = 5.0
    MAX_SPEED = 30
    # MERGE_POINT_X = -50
    # CAR_LENGTH = 5.0
    # USE_ALTERNATE_TRAFFIC_DISTRIBUTION = False
    # USE_SIMPLE_TRAFFIC_DISTRIBUTION = True
    # TRAFFIC_DENSITY = "low"  # "low" or "medium" or "high"

    # Simple traffic distribution
    # VARY_TRAFFIC_START_TIMES = True
    # BASE_TRAFFIC_INTERVAL = 1.2
    # OTHER_CAR_SPEED = 7.0



    # Random start speed
    START_SPEED = 15
    RANDOMIZE_START_SPEED = True
    START_SPEED_VARIANCE = 5
    MIN_START_SPEED = 5
    MAX_START_SPEED = 25








    # Tabular RL
    JERK_VALUES = {0: -5, 1: -2.5, 2: 0, 3: 2.5, 4: 5}
    TRAINING_TICK_LENGTH = 0.2
    MAX_EPISODE_LENGTH = 100
    STEP_SIZE = 0.01
    GAMMA = 1.0
    NUM_TRAINING_EPISODES = 150000
    STEP_SIZE_HALF_PER_EPISODES = 20000
    # Parameters for evaluation
    EVALUATION_PERIOD = 2000            # Evaluate after this many training episodes
    NUM_EVALUATION_EPISODES = 100       # Evaluate over this many episodes
    EVALUATION_EPISODE_LENGTH = 50      # Evaluation epsiode length in seconds
    EVALUATION_TICK_LENGTH = 0.2
    AVOID_UNVISITED_STATES = True





