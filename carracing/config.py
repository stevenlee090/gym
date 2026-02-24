"""
Experiment configuration for CarRacing-v3 SAC training (continuous action space).
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
TOTAL_TIMESTEPS = 1_500_000
N_ENVS = 8                      # More envs now that GPU handles gradient updates
SEED = 42

# SAC hyperparameters (tuned for pixel-based continuous control)
SAC_KWARGS = {
    "policy": "CnnPolicy",
    "learning_rate": 3e-4,
    "buffer_size": 300_000,     # Can afford more now images are smaller (64x64 grayscale)
    "learning_starts": 10_000,  # Collect random data before first update
    "batch_size": 256,
    "gamma": 0.99,
    "tau": 0.005,
    "ent_coef": "auto",         # Auto-tune entropy for exploration
    "train_freq": 1,
    "gradient_steps": 1,
    "optimize_memory_usage": False,
}

# Callbacks
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 10
CHECKPOINT_FREQ = 100_000

# Paths
LOG_DIR = "logs"
MODEL_DIR = "models"
VIDEO_DIR = "videos"
MODEL_NAME = "sac_carracing"
