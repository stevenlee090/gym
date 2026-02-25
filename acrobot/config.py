"""
Experiment configuration for Acrobot-v1 PPO training.
"""

# Environment
ENV_ID = "Acrobot-v1"
# No special kwargs — Acrobot-v1 uses default dynamics.

# Training
TOTAL_TIMESTEPS = 500_000
N_ENVS = 16                 # Parallel environments
SEED = 42

# PPO hyperparameters (tuned for Acrobot; reference: RL Baselines3 Zoo)
PPO_KWARGS = {
    "learning_rate": 1e-3,
    "n_steps": 256,         # Steps per env before update
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.94,
    "clip_range": 0.2,
    "ent_coef": 0.0,        # Acrobot doesn't need extra entropy bonus
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy": "MlpPolicy",
}

# Callbacks
EVAL_FREQ = 10_000          # Evaluate every N timesteps
N_EVAL_EPISODES = 20        # Episodes per evaluation
CHECKPOINT_FREQ = 50_000    # Save checkpoint every N timesteps

# Paths
LOG_DIR = "logs"
MODEL_DIR = "models"
VIDEO_DIR = "videos"
MODEL_NAME = "ppo_acrobot"
