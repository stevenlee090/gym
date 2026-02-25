"""
Experiment configuration for MountainCarContinuous-v0 PPO training.

Why PPO instead of TD3/SAC?
  Both TD3 and SAC failed on this sparse-reward environment: SAC's entropy
  collapsed (stopped exploring), and TD3 got stuck in a constant-force local
  optimum that never built rocking momentum.

  PPO with kinetic energy reward shaping (r += 0.1 * velocity²) is a reliable
  solution — the agent is rewarded for building speed in any direction, which
  directly incentivises the rocking behaviour needed to crest the hill.

Why kinetic energy shaping?
  Φ = velocity² gives a small dense signal every step. The car must rock
  (alternate directions) to build speed, so the agent naturally discovers the
  goal without any luck-based exploration.
"""

# Environment
ENV_ID = "MountainCarContinuous-v0"

# Training
TOTAL_TIMESTEPS = 200_000
N_ENVS = 8
SEED = 42

# Reward shaping
SHAPING_WEIGHT = 100.0      # r' = r_env + SHAPING_WEIGHT * velocity²
                             # Needs to be large: v_max=0.07, so w*v²_max=0.49/step
                             # vs action penalty of -0.1/step at action=1.0

# PPO hyperparameters
PPO_KWARGS = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,        # Entropy bonus prevents early policy collapse
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy": "MlpPolicy",
}

# Callbacks
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 10
CHECKPOINT_FREQ = 50_000

# Paths
LOG_DIR = "logs"
MODEL_DIR = "models"
VIDEO_DIR = "videos"
MODEL_NAME = "ppo_mountaincar"
