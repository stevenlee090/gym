# MountainCarContinuous-v0 TD3 Experiment

Reinforcement learning experiment training a TD3 agent on the [MountainCarContinuous-v0](https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/) Gymnasium environment.

## Environment

| Property | Value |
|---|---|
| Observation space | Box(2,) — car position and velocity |
| Action space | Box(1,) — continuous force in [−1, 1] |
| Reward | −0.1 × action² per step; +100 on reaching the goal |
| Solved threshold | Mean reward ≥ 90 |
| Max episode steps | 999 |

The goal is to drive an underpowered car up a steep hill by building momentum from a valley. The car's engine alone isn't strong enough — it must rock back and forth first.

### Why TD3 instead of SAC or PPO?

This environment has **sparse rewards** (+100 only on reaching the goal, −0.1×action² per step). SAC's entropy auto-tuning collapses early — the agent learns to output near-zero actions to minimise the step penalty and never discovers the goal. TD3 injects **Gaussian noise (σ=0.1) into every training action**, guaranteeing sustained exploration regardless of what the policy learns.

## Setup

```bash
uv sync  # from repo root
```

## Training

```bash
cd mountaincar
uv run python train.py
```

Options:
```
--timesteps INT    Total training steps (default: 300_000)
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

> `<run_name>` is printed at the start of training (e.g. `sac_mountaincar_seed42`). `best_model` is saved whenever eval score improves.

## Monitoring

```bash
uv run tensorboard --logdir logs/tensorboard
```

## Project Structure

```
mountaincar/
├── config.py       # Hyperparameters and paths
├── train.py        # SAC training script
├── evaluate.py     # Evaluation + video recording
├── logs/           # TensorBoard logs (git-ignored)
├── models/         # Saved checkpoints + best model (git-ignored)
└── videos/         # Recorded evaluation videos (git-ignored)
```

## Algorithm: TD3

Twin Delayed Deep Deterministic Policy Gradient with key hyperparameters:

| Param | Value |
|---|---|
| Learning rate | 1e-3 |
| Batch size | 100 |
| Learning starts | 1000 |
| Action noise | N(0, 0.1) |
| Gamma | 0.99 |
| Tau | 0.005 |
| Policy delay | 2 |
