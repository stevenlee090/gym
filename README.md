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
| [`carracing/`](carracing/) | [CarRacing-v3](https://gymnasium.farama.org/environments/box2d/car_racing/) | PPO | Continuous Box(3) | ≥ 900 |
| [`acrobot/`](acrobot/) | [Acrobot-v1](https://gymnasium.farama.org/environments/classic_control/acrobot/) | PPO | Discrete(3) | ≥ −100 |
| [`mountaincar/`](mountaincar/) | [MountainCarContinuous-v0](https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/) | PPO + KE shaping | Continuous Box(1) | ≥ 90 |

## Quick Start

```bash
# LunarLander
cd lunarlander && uv run python train.py

# CarRacing
cd carracing && uv run python train.py

# Acrobot
cd acrobot && uv run python train.py

# MountainCar (continuous)
cd mountaincar && uv run python train.py
```

## Watch a Trained Agent

After training, models are saved under `models/<run_name>/`. Use `best_model` for the best checkpoint:

```bash
cd lunarlander && uv run python evaluate.py --model models/<run_name>/best_model --render
cd carracing  && uv run python evaluate.py --model models/<run_name>/best_model --render
cd acrobot    && uv run python evaluate.py --model models/<run_name>/best_model --render
cd mountaincar && uv run python evaluate.py --model models/<run_name>/best_model --render
```

## Monitoring

```bash
uv run tensorboard --logdir <folder>/logs/tensorboard
```

## Notes

- **MountainCarContinuous** uses a `KineticEnergyShapingWrapper` (`r' = r + 100·v²`) to overcome the sparse reward problem — SAC and TD3 both fail on this environment without it.
- All MlpPolicy experiments run on CPU (transfer overhead makes MPS/CUDA slower for small networks).
- CarRacing uses CnnPolicy on MPS (GPU pays off for pixel-based observations).
