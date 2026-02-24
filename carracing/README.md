# CarRacing-v3 SAC Experiment

Reinforcement learning experiment training a SAC agent on the [CarRacing-v3](https://gymnasium.farama.org/environments/box2d/car_racing/) Gymnasium environment using continuous actions and pixel observations.

## Environment

| Property | Value |
|---|---|
| Observation space | Box(96, 96, 3) — RGB pixel image |
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
--timesteps INT    Total training steps (default: 1_500_000)
--n-envs    INT    Parallel environments (default: 4)
--seed      INT    Random seed (default: 42)
--resume    PATH   Resume from checkpoint
```

## Evaluation

**Watch live:**
```bash
uv run python evaluate.py --model models/sac_carracing_seed42/best_model --render
```

**Save a video:**
```bash
uv run python evaluate.py --model models/sac_carracing_seed42/best_model --record
```

**Stats only:**
```bash
uv run python evaluate.py --model models/sac_carracing_seed42/best_model --episodes 10
```

## Monitoring

```bash
uv run tensorboard --logdir logs/tensorboard
```

## Project Structure

```
carracing/
├── config.py       # Hyperparameters and paths
├── train.py        # SAC training script
├── evaluate.py     # Evaluation + video recording
├── logs/           # TensorBoard logs (git-ignored)
├── models/         # Saved checkpoints + best model (git-ignored)
└── videos/         # Recorded evaluation videos (git-ignored)
```

## Algorithm: SAC

Soft Actor-Critic with CNN policy for pixel observations.

| Param | Value |
|---|---|
| Policy | CnnPolicy |
| Learning rate | 3e-4 |
| Buffer size | 200,000 |
| Learning starts | 10,000 |
| Batch size | 256 |
| Gamma | 0.99 |
| Tau | 0.005 |
| Entropy coef | auto |
| Frame stack | 4 |

SAC is off-policy (replay buffer) and maximizes entropy to encourage exploration — well suited for the continuous steering/gas/brake action space.
