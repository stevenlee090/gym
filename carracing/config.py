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

# Observation preprocessing
OBS_GRAYSCALE = True            # 96x96x3 → 96x96x1
OBS_RESIZE = (64, 64)           # 96x96 → 64x64

# Device — CnnPolicy benefits significantly from MPS on Apple Silicon
DEVICE = "mps"

# Training
TOTAL_TIMESTEPS = 5_000_000     # More steps — previous run peaked at 1.1M then degraded
N_ENVS = 8                      # Reduced from 16 — less variance per batch
SEED = 42

# Schedules — both LR and clip_range decay linearly to prevent late-training instability
LR_INIT = 3e-4
CLIP_RANGE_INIT = 0.1           # Tighter than default 0.2 (35% clip fraction was too high)

# PPO hyperparameters
PPO_KWARGS = {
    "policy": "CnnPolicy",
    "n_steps": 1024,            # Larger rollout (batch = 1024×8 = 8192, same as before)
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ent_coef": 0.005,          # Lower than before (0.01 kept std too high late in training)
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": 0.015,         # Early-stop updates if policy changes too much — key fix
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
