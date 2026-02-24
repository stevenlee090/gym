# LunarLander-v3 PPO Experiment

Reinforcement learning experiment training a PPO agent on the [LunarLander-v3](https://gymnasium.farama.org/environments/box2d/lunar_lander/) Gymnasium environment.

## Environment

| Property | Value |
|---|---|
| Observation space | Box(8,) — position, velocity, angle, leg contacts |
| Action space | Discrete(4) — nothing, left engine, main engine, right engine |
| Solved threshold | Mean reward ≥ 200 |

## Setup

```bash
uv sync  # from repo root
```

## Training

```bash
cd lunarlander
uv run python train.py
```

Options:
```
--timesteps INT    Total training steps (default: 1_000_000)
--n-envs    INT    Parallel environments (default: 16)
--seed      INT    Random seed (default: 42)
--resume    PATH   Resume from checkpoint
--device    STR    Device override: cpu, mps, cuda (auto-detected)
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

> `<run_name>` is printed at the start of training (e.g. `ppo_lunarlander_seed42`). `best_model` is saved whenever eval score improves, so it's available before training finishes.


## Monitoring

```bash
uv run tensorboard --logdir logs/tensorboard
```

## Project Structure

```
lunarlander/
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
| Learning rate | 3e-4 |
| n_steps | 1024 |
| batch_size | 64 |
| n_epochs | 10 |
| gamma | 0.999 |
| gae_lambda | 0.98 |
| ent_coef | 0.01 |
