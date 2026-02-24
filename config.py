"""
Experiment configuration for LunarLander-v3 PPO training.
"""

# Environment
ENV_ID = "LunarLander-v3"
ENV_KWARGS = {
    "continuous": False,   # Discrete action space (4 actions)
    "gravity": -10.0,
    "enable_wind": False,
    "wind_power": 15.0,
    "turbulence_power": 1.5,
}

# Training
TOTAL_TIMESTEPS = 1_000_000
N_ENVS = 8                  # Parallel environments for data collection
SEED = 42

# PPO hyperparameters (tuned for LunarLander)
PPO_KWARGS = {
    "learning_rate": 3e-4,
    "n_steps": 1024,        # Steps per env before update
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.999,
    "gae_lambda": 0.98,
    "clip_range": 0.2,
    "ent_coef": 0.01,       # Entropy bonus to encourage exploration
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy": "MlpPolicy",
}

# Callbacks
EVAL_FREQ = 10_000          # Evaluate every N timesteps
N_EVAL_EPISODES = 20        # Episodes per evaluation
CHECKPOINT_FREQ = 100_000   # Save checkpoint every N timesteps

# Paths
LOG_DIR = "logs"
MODEL_DIR = "models"
VIDEO_DIR = "videos"
MODEL_NAME = "ppo_lunarlander"
