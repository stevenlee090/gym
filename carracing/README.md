# CarRacing-v3 PPO Experiment

Reinforcement learning experiment training a PPO agent on the [CarRacing-v3](https://gymnasium.farama.org/environments/box2d/car_racing/) Gymnasium environment using continuous actions and pixel observations.

## Environment

| Property | Value |
|---|---|
| Observation space | Box(96, 96, 3) → preprocessed to 64×64 grayscale |
| Action space | Box(3,) continuous — [steering, gas, brake] |
| Frame stacking | 4 frames (needed to infer velocity) |
| Solved threshold | Mean reward ≥ 900 |

## Setup

```bash
uv sync  # from repo root
```

## Training

```bash
cd carracing
uv run python train.py
```

Options:
```
--timesteps INT    Total training steps (default: 3_000_000)
--n-envs    INT    Parallel environments (default: 16)
--seed      INT    Random seed (default: 42)
--resume    PATH   Resume from checkpoint
--device    STR    Device override: mps, cpu, cuda
```

## Watch the Agent Drive

**Watch the best saved model live:**
```bash
cd carracing
uv run python evaluate.py --model models/ppo_carracing_seed42/best_model --render
```

> `best_model` is updated automatically during training whenever eval score improves — you can watch it mid-training.

**Save a video:**
```bash
uv run python evaluate.py --model models/ppo_carracing_seed42/best_model --record
```

**Stats only (no window):**
```bash
uv run python evaluate.py --model models/ppo_carracing_seed42/best_model --episodes 10
```

## Monitoring

```bash
uv run tensorboard --logdir logs/tensorboard
```

## Project Structure

```
carracing/
├── config.py       # Hyperparameters and paths
├── train.py        # PPO training script
├── evaluate.py     # Evaluation + video recording
├── wrappers.py     # PIL-based grayscale + resize (avoids cv2/SDL2 conflict)
├── logs/           # TensorBoard logs (git-ignored)
├── models/         # Saved checkpoints + best model (git-ignored)
└── videos/         # Recorded evaluation videos (git-ignored)
```

## Algorithm: PPO

Proximal Policy Optimization with CNN policy for pixel observations.

| Param | Value |
|---|---|
| Policy | CnnPolicy |
| Learning rate | 3e-4 |
| n_steps | 512 |
| Batch size | 256 |
| n_epochs | 10 |
| Gamma | 0.99 |
| gae_lambda | 0.95 |
| ent_coef | 0.01 |
| Frame stack | 4 |
| Obs size | 64×64 grayscale |
| Device | MPS (Apple Silicon) |

PPO is on-policy with explicit entropy regularisation (`ent_coef=0.01`) to prevent the exploration collapse that off-policy methods like SAC can suffer from on pixel tasks. Large batches (512×16=8192 samples) make efficient use of the MPS GPU.
