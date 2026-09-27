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

### Go1 joystick (brax/, GPU)
- Separate uv project (`cd brax && uv sync`): JAX CUDA + brax + MuJoCo Playground; jax pinned <0.10 (brax 0.14 calls removed `jax.device_put_replicated`)
- Brax PPO, 8,192 envs on an RTX 3080 (WSL2): 124.5M steps in 59 min, eval reward 27.2
- Learning order: stand -> turn in place (26-52M) -> walk/run/strafe (~60M+)
- `export_web.py` strips the model to 31 KB and checks sim-to-sim in CPU MuJoCo; `web/sim.js` runs it live in the browser
- Public demo: https://claude.ai/artifact/X8f2sowZR5WdLM6BP4ZozQ
- WSL2: MuJoCo rendering needs `MUJOCO_GL=egl`; TensorBoard at http://localhost:6006 with `--bind_all`
