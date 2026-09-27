# Gym Project Memory

## Project Structure
- Root: `/Users/stevenlee/github/gym`
- Package manager: `uv` (always use `uv run python` or `uv sync`)
- Each environment lives in its own subfolder: `lunarlander/`, `carracing/`, `acrobot/`
- Each subfolder has: `config.py`, `train.py`, `evaluate.py`, `README.md`
- Shared root `pyproject.toml` with `gymnasium[box2d]`, `stable-baselines3`, `torch`, etc.

## Conventions
- Algorithm: PPO (stable-baselines3) with `MlpPolicy` for vector obs environments
- Device: always `"cpu"` for MlpPolicy (MPS/CUDA overhead not worth it for small nets)
- N_ENVS: 16 parallel environments
- Logging: TensorBoard under `logs/tensorboard/`
- Checkpoints + best model saved under `models/<run_name>/`
- Solve threshold printed in evaluate.py output

## Environments

### LunarLander-v3
- Obs: Box(8,), Action: Discrete(4)
- Solved at mean ≥ 200
- Timesteps: 1_000_000

### Acrobot-v1
- Obs: Box(6,) — cos/sin of joint angles + angular velocities
- Action: Discrete(3) — torques −1, 0, +1
- Solved at mean ≥ −100; agent achieved ~−67 in 500k steps (~2.5 min)
- Timesteps: 500_000, lr=1e-3, n_steps=256, n_epochs=10, gamma=0.99, gae_lambda=0.94

### CarRacing (PPO)
- Uses PPO (switched from SAC)
- Has a `wrappers.py` (likely frame stacking / grayscale)
