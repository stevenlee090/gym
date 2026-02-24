# LunarLander-v3 PPO Experiment

Reinforcement learning experiment training a PPO agent on the [LunarLander-v3](https://gymnasium.farama.org/environments/box2d/lunar_lander/) Gymnasium environment.

## Environment

| Property | Value |
|---|---|
| Observation space | Box(8,) — position, velocity, angle, leg contacts |
| Action space | Discrete(4) — nothing, left engine, main engine, right engine |
| Solved threshold | Mean reward ≥ 200 over 100 episodes |

## Setup

```bash
uv sync
```

## Training

```bash
uv run python train.py
```

Options:
```
--timesteps INT    Total training steps (default: 1_000_000)
--n-envs    INT    Parallel environments (default: 8)
--seed      INT    Random seed (default: 42)
--resume    PATH   Resume from checkpoint
```

## Evaluation

```bash
uv run python evaluate.py --model models/ppo_lunarlander_seed42/best_model
```

Options:
```
--episodes INT   Number of evaluation episodes (default: 20)
--render         Watch the agent live in a pygame window
--record         Record a video to videos/
```

**Watch live:**
```bash
uv run python evaluate.py --model models/ppo_lunarlander_seed42/best_model --render
```

**Save a video:**
```bash
uv run python evaluate.py --model models/ppo_lunarlander_seed42/best_model --record
```

> `best_model` is saved whenever eval score improves during training, so it's available before training finishes.

## Monitoring

```bash
uv run tensorboard --logdir logs/tensorboard
```

## Project Structure

```
gym/
├── config.py       # Hyperparameters and paths
├── train.py        # Training script
├── evaluate.py     # Evaluation + video recording
├── pyproject.toml  # Dependencies (managed by uv)
├── logs/           # TensorBoard logs + eval logs (git-ignored)
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
