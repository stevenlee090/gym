# Acrobot-v1 PPO Experiment

Reinforcement learning experiment training a PPO agent on the [Acrobot-v1](https://gymnasium.farama.org/environments/classic_control/acrobot/) Gymnasium environment.

## Environment

| Property | Value |
|---|---|
| Observation space | Box(6,) — cos/sin of 2 joint angles + angular velocities |
| Action space | Discrete(3) — apply torque −1, 0, or +1 to the actuated joint |
| Reward | −1 per step; 0 on success |
| Solved threshold | Mean reward ≥ −100 |
| Max episode steps | 500 |

The goal is to swing the end of a two-link chain above the fixed link within as few steps as possible.

## Setup

```bash
uv sync  # from repo root
```

## Training

```bash
cd acrobot
uv run python train.py
```

Options:
```
--timesteps INT    Total training steps (default: 500_000)
--n-envs    INT    Parallel environments (default: 16)
--seed      INT    Random seed (default: 42)
--resume    PATH   Resume from checkpoint
```

## Evaluation

**Watch live:**
```bash
uv run python evaluate.py --model models/<run_name>/best_model --render
```

**Save a video:**
```bash
uv run python evaluate.py --model models/<run_name>/best_model --record
```

> `<run_name>` is printed at the start of training (e.g. `ppo_acrobot_seed42`). `best_model` is saved whenever eval score improves, so it's available before training finishes.

## Monitoring

```bash
uv run tensorboard --logdir logs/tensorboard
```

## Project Structure

```
acrobot/
├── config.py       # Hyperparameters and paths
├── train.py        # PPO training script
├── evaluate.py     # Evaluation + video recording
├── logs/           # TensorBoard logs (git-ignored)
├── models/         # Saved checkpoints + best model (git-ignored)
└── videos/         # Recorded evaluation videos (git-ignored)
```

## Algorithm: PPO

Proximal Policy Optimization with key hyperparameters:

| Param | Value |
|---|---|
| Learning rate | 1e-3 |
| n_steps | 256 |
| batch_size | 64 |
| n_epochs | 10 |
| gamma | 0.99 |
| gae_lambda | 0.94 |
| ent_coef | 0.0 |
