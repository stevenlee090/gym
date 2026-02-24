"""
Experiment configuration for CarRacing-v3 PPO training (continuous action space).
"""

# Environment
ENV_ID = "CarRacing-v3"
ENV_KWARGS = {
    "continuous": True,         # Box(3,): [steering, gas, brake]
    "domain_randomize": False,
}
N_STACK = 4                     # Frame stacking — needed to infer velocity from pixels

# Observation preprocessing (reduces CNN input ~7x vs raw 96x96x3)
OBS_GRAYSCALE = True            # 96x96x3 → 96x96x1
OBS_RESIZE = (64, 64)           # 96x96 → 64x64

# Device — CnnPolicy benefits significantly from MPS on Apple Silicon
DEVICE = "mps"

# Training
TOTAL_TIMESTEPS = 3_000_000     # PPO needs more steps than SAC (on-policy)
N_ENVS = 16                     # PPO scales well with many envs
SEED = 42

# PPO hyperparameters (tuned for pixel-based continuous control)
PPO_KWARGS = {
    "policy": "CnnPolicy",
    "learning_rate": 3e-4,
    "n_steps": 512,             # Steps per env before update (total batch = 512×16 = 8192)
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,           # Explicit entropy — prevents the collapse we saw with SAC
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}

# Callbacks
EVAL_FREQ = 20_000
N_EVAL_EPISODES = 10
CHECKPOINT_FREQ = 200_000

# Paths
LOG_DIR = "logs"
MODEL_DIR = "models"
VIDEO_DIR = "videos"
MODEL_NAME = "ppo_carracing"
