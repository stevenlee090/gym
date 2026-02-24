# gym

Reinforcement learning experiments using [Gymnasium](https://gymnasium.farama.org), managed with [uv](https://docs.astral.sh/uv/).

## Setup

```bash
uv sync
```

## Experiments

| Folder | Environment | Algorithm | Action Space | Solved |
|---|---|---|---|---|
| [`lunarlander/`](lunarlander/) | [LunarLander-v3](https://gymnasium.farama.org/environments/box2d/lunar_lander/) | PPO | Discrete(4) | ≥ 200 |
| [`carracing/`](carracing/) | [CarRacing-v3](https://gymnasium.farama.org/environments/box2d/car_racing/) | SAC | Continuous Box(3) | ≥ 900 |

## Quick Start

```bash
# LunarLander
cd lunarlander && uv run python train.py

# CarRacing
cd carracing && uv run python train.py
```

## Monitoring

```bash
uv run tensorboard --logdir lunarlander/logs/tensorboard
uv run tensorboard --logdir carracing/logs/tensorboard
```
