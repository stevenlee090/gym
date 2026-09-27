# Go1 Joystick — GPU PPO with MuJoCo Playground

A Unitree Go1 quadruped learns to follow joystick commands (forward speed, sideways speed, turn rate) from scratch. Unlike the other folders, **the physics runs on the GPU too**: [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) steps 8,192 robots in one batched MJX call and [Brax](https://github.com/google/brax) PPO trains on them without the data ever leaving the GPU.

**[Live demo](https://claude.ai/artifact/X8f2sowZR5WdLM6BP4ZozQ)**: drive the trained policy in your browser (WASD, Q/E, Shift), then see recorded checkpoints and training curves.

This is its own uv project, separate from the root one, so JAX's CUDA libraries don't clash with PyTorch's.

## How it works

| Piece | Details |
|---|---|
| Task | `Go1JoystickFlatTerrain`. A new random command is sampled every ~5 s. The reward pays for tracking the commanded linear and angular velocity and penalises tilt, jerky actions, energy use, foot slip and dragging feet. 1,000-step episodes at 50 Hz control (20 s). |
| Observations | Policy: 48 noisy "real sensor" values (gyro, gravity vector, body-velocity estimate, joint angles/velocities, last action, command). Value function: 123 values including privileged simulator state (motor forces, foot contacts, foot velocities). |
| Algorithm | Brax PPO with Playground's tuned hyperparameters: 8,192 envs, unroll 20, 32 minibatches × 4 epochs, γ = 0.97, lr 3e-4, entropy 1e-2. Policy and value MLPs are both 512-256-128. |
| Robustness | Domain randomisation per robot: floor friction, joint friction and armature, link masses and centre of mass. Disable with `--no-dr`. |

## Results (RTX 3080, seed 0)

124.5M environment steps in 59 minutes (~36k steps/s median, 8,192 robots in parallel; the first ~4 min is JAX compiling). Evaluation reward rose from 0 to **27.2 ± 3.0**, and the final policy survives 99% of 20-second random-command episodes.

| Checkpoint | Training time | Reward (128 random-command episodes) | Speed error | Turn error |
|---|---|---|---|---|
| 6.6M steps | 4 min | 16.7 ± 5.5 | 0.80 m/s | 0.59 rad/s |
| 26M steps | 13 min | 20.6 ± 4.8 | 0.80 m/s | 0.14 rad/s |
| 52M steps | 23 min | 22.4 ± 4.8 | 0.78 m/s | 0.14 rad/s |
| 125M steps | 58 min | 29.8 ± 3.6 | 0.17 m/s | 0.14 rad/s |

Errors are RMS gaps between the commanded and measured body velocity over the scripted demo. The policy learns in a clear order: first stand (6.6M), then **turn on the spot** (26M and 52M both track turn commands but ignore forward and sideways commands), and only after ~60M steps walk, run, strafe and reverse. The final policy tracks 1.0 m/s forward at 0.96 m/s and a 1.5 m/s run at 1.44 m/s.

## Setup

```bash
cd brax
uv sync
```

Needs an NVIDIA GPU with a recent driver. `jax` is pinned below 0.10 because brax 0.14 still calls `jax.device_put_replicated`, which JAX 0.10 removed.

## Usage

```bash
# Train (100M steps requested, ~125M actual after epoch rounding; ~1 h on an RTX 3080)
uv run python train.py
uv run python train.py --timesteps 200000000        # Playground's full budget
uv run python train.py --env Go1JoystickRoughTerrain

# Evaluate the latest checkpoint: scores 128 random-command episodes, then renders
# the scripted joystick demo to videos/<env>_<tag>.mp4 plus a velocity trace (.json)
uv run python evaluate.py --run models/<run_name>

# Render an earlier checkpoint for comparison
uv run python evaluate.py --run models/<run_name> --checkpoint 000026214400

# Export the policy for the in-browser simulator (web/) and check it in CPU MuJoCo
uv run python export_web.py --run models/<run_name> --check

# Build the dashboard (dashboard.html) from every rendered demo and the training logs
uv run python visualise.py --run models/<run_name> --open

# Serve it (the live simulator needs http://, not file://)
python3 -m http.server 8000     # then open http://localhost:8000/dashboard.html
```

Checkpoints are saved under `models/<run_name>/checkpoints/<step>/` at every evaluation (20 per run). The scripted demo is `DEMO_COMMANDS` in [`config.py`](config.py): stand, walk, run, turn left/right, strafe, reverse, spin.

## Dashboard

`visualise.py` builds a single page, following the same pattern as `game2048/visualise.py`:

- **Drive it yourself:** the trained policy running live in the browser. Use W/S for forward and back, A/D to sidestep, Q/E to turn, Shift to run, F to shove and R to reset. There are also on-screen keys for touch screens. See below.
- The recorded demo video synced to a readout of the joystick command vs the robot's actual body velocity, with a top-down velocity dial and a turn-rate gauge.
- Forward, sideways and turn tracking strips with a playhead. Click or drag to seek.
- A checkpoint picker and table (reward, survival, tracking error) to watch the gait develop.
- Evaluation reward, simulator throughput, and a per-term reward breakdown (first vs latest evaluation).

The page loads `videos/*.mp4` by relative path, so keep it in this folder. It works mid-training, since `train.py` rewrites `models/<run_name>/metrics.json` after every log.

## In-browser simulator

[`web/sim.js`](web/sim.js) runs the policy with no server:

- **Physics:** the official MuJoCo WebAssembly build ([`@mujoco/mujoco`](https://www.npmjs.com/package/@mujoco/mujoco), same version as the Python package), vendored into `web/vendor/` by `export_web.py`.
- **Model:** `web/go1.mjb` is the exact training model (PD gains, 4 ms timestep, one solver iteration) with its visual-only geoms, meshes and skybox stripped: 31 KB instead of 49 MB. Only the floor and foot spheres collide, and every link has explicit inertia, so the dynamics are unchanged. `export_web.py` checks this field by field.
- **Policy:** the observation normaliser and 48→512→256→128→24 MLP in plain JavaScript. It self-checks against a NumPy reference action on load (difference ~1e-7).
- **Rendering:** three.js. The robot meshes (2.5 MB) and the compiled model travel base64-encoded in `web/go1_scene.json`, because artifact hosting serves JSON but not raw binaries.

The browser runs standard CPU MuJoCo while training used MJX on the GPU, so this is a small sim-to-sim transfer. `export_web.py --check` runs the same maths in Python first. For the final checkpoint it tracked forward 1.0 → 1.01 m/s, backward 0.8 → 0.76, strafe 0.6 → 0.56 and turn 1.0 → 0.94 rad/s, with no falls.

## Monitoring

```bash
uv run tensorboard --logdir logs/tensorboard --bind_all
```

On WSL2, open http://localhost:6006 in the Windows browser. `train/*` updates every ~1M steps (reward, steps/s, each reward term); `eval/*` updates at each of the 20 evaluations.

`eval/episode_reward` looks small (~20) because Playground multiplies the summed reward terms by the 20 ms timestep. The per-term curves (`eval/episode_reward/tracking_lin_vel`, `.../feet_air_time`, ...) show more clearly what the policy is learning.

## Notes and gotchas

- **Rendering on WSL2** uses EGL (`MUJOCO_GL=egl`, set by default in the scripts). A `TypeError: 'NoneType' object is not callable` from `mujoco/egl` at interpreter exit is harmless teardown noise.
- **"solver iterations limit reached" warnings** are expected. Playground runs Go1 with one constraint-solver iteration for speed.
- **Short runs overshoot `--timesteps`.** Brax runs at least one full training epoch (~1.6M steps with these settings) between evaluations, so `--timesteps 2000000` with 20 evaluations trains for ~32M steps.
- **Checkpoint loading** needs a small patch for brax 0.14, which saves unset kernel initialisers as `null` and then fails to load them. `evaluate.py` applies it.
- **Restarting from a stand.** Mid-training checkpoints (seen up to 92M steps) would not start walking on a pure forward or sideways command after standing still with a zero command. A turn command got them moving. The final checkpoint starts walking from a standstill without help, and the demo script puts "stand still" last so every checkpoint is judged on walking.
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` stops JAX reserving 75% of VRAM up front, so evaluation can share the GPU with a training run.

## Ideas to push further

- **Rough terrain:** `--env Go1JoystickRoughTerrain` (heightfield floor), ideally warm-started from the flat policy.
- **Pushes:** enable `pert_config` in the env config to train recovery from random shoves.
- **Other skills:** `Go1Handstand`, `Go1Getup`, or other robots such as `BerkeleyHumanoidJoystickFlatTerrain` and `SpotJoystickGaitTracking`.
- **Sim-to-real:** export the policy to ONNX; Playground's policies transfer to the real Go1.
