"""
Evaluate a trained Playground PPO policy and render a scripted joystick demo.

The demo drives the robot through config.DEMO_COMMANDS (forward, turn, strafe, ...)
and records the commanded vs. actual body velocity alongside the video.

Usage:
    python evaluate.py --run models/ppo_Go1JoystickFlatTerrain_seed0
    python evaluate.py --run models/<run_name> --checkpoint 000049152000 --tag mid
    python evaluate.py --run models/<run_name> --episodes 256 --no-video
"""

import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore")

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import jax
import jax.numpy as jnp
import numpy as np
from brax.training import networks as brax_networks
from brax.training.agents.ppo import checkpoint
from mujoco_playground import registry

import config

# brax 0.14 saves unset kernel initialisers as null, then fails to look `None` up on load.
brax_networks.KERNEL_INITIALIZER.setdefault(None, None)


def latest_checkpoint(run_dir: str) -> str:
    ckpt_root = os.path.join(run_dir, "checkpoints")
    steps = sorted(d for d in os.listdir(ckpt_root) if d.isdigit())
    if not steps:
        raise FileNotFoundError(f"No checkpoints in {ckpt_root}")
    return os.path.join(ckpt_root, steps[-1])


def load_env(env_name: str, n_envs: int):
    env_cfg = registry.get_default_config(env_name)
    env_cfg.naconmax = 4 * max(n_envs, 1)  # contact buffer sized for this batch
    return registry.load(env_name, config=env_cfg)


def evaluate_random_commands(env_name, policy, n_envs, seed):
    """Average episode return over n_envs parallel episodes with random commands."""
    env = load_env(env_name, n_envs)
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))
    act = jax.jit(jax.vmap(policy))

    keys = jax.random.split(jax.random.PRNGKey(seed), n_envs)
    state = reset(keys)
    total = jnp.zeros(n_envs)
    alive = jnp.ones(n_envs)
    for _ in range(env._config.episode_length):
        keys = jax.vmap(lambda k: jax.random.split(k)[0])(keys)
        action, _ = act(state.obs, keys)
        state = step(state, action)
        total += state.reward * alive
        alive *= 1.0 - state.done
    total = np.asarray(total)
    return float(total.mean()), float(total.std()), float(np.asarray(alive).mean())


def run_demo(env_name, policy, seed):
    """Drive a single robot through the scripted command sequence."""
    env = load_env(env_name, 1)
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    act = jax.jit(policy)

    key = jax.random.PRNGKey(seed)
    state = reset(key)
    state.info["steps_until_next_cmd"] = jnp.array(10**9, dtype=jnp.int32)

    trajectory, trace, segments = [], [], []
    t = 0
    for label, cmd, seconds in config.DEMO_COMMANDS:
        n_steps = int(round(seconds / env.dt))
        segments.append({"label": label, "command": cmd,
                         "start_s": round(t * env.dt, 3),
                         "end_s": round((t + n_steps) * env.dt, 3)})
        command = jnp.array(cmd, dtype=jnp.float32)
        for _ in range(n_steps):
            state.info["command"] = command
            key, act_key = jax.random.split(key)
            action, _ = act(state.obs, act_key)
            state = step(state, action)
            state.info["steps_until_next_cmd"] = jnp.array(10**9, dtype=jnp.int32)
            trajectory.append(state)
            vel = np.asarray(env.get_local_linvel(state.data))
            gyro = np.asarray(env.get_gyro(state.data))
            trace.append({
                "t": round(t * env.dt, 3),
                "cmd": [float(c) for c in cmd],
                "vx": round(float(vel[0]), 4),
                "vy": round(float(vel[1]), 4),
                "wz": round(float(gyro[2]), 4),
                "fell": bool(state.done),
            })
            t += 1
            if bool(state.done):
                break
        if bool(state.done):
            print(f"  robot fell during '{label}'")
            break

    return env, trajectory, trace, segments


def evaluate(args):
    ckpt = args.checkpoint
    ckpt_path = (os.path.join(args.run, "checkpoints", ckpt) if ckpt
                 else latest_checkpoint(args.run))
    ckpt_path = os.path.abspath(ckpt_path)
    step = int(os.path.basename(ckpt_path))
    tag = args.tag or f"step{step}"
    print(f"\nLoading policy: {ckpt_path}  ({step:,} env steps)")
    policy = checkpoint.load_policy(ckpt_path, deterministic=True)

    results = {"checkpoint": os.path.basename(ckpt_path), "env_steps": step}

    if args.episodes > 0:
        mean, std, survival = evaluate_random_commands(
            args.env, policy, args.episodes, args.seed)
        print(f"  random commands over {args.episodes} episodes: "
              f"reward {mean:.2f} ± {std:.2f}, survival {survival:.0%}")
        results.update(reward_mean=mean, reward_std=std, survival=survival)

    if args.video:
        env, trajectory, trace, segments = run_demo(args.env, policy, args.seed)
        os.makedirs(config.VIDEO_DIR, exist_ok=True)
        video_path = os.path.join(config.VIDEO_DIR, f"{args.env}_{tag}.mp4")
        skip = config.RENDER_FRAME_SKIP
        frames = env.render(trajectory[::skip], height=config.RENDER_HEIGHT,
                            width=config.RENDER_WIDTH, camera=config.RENDER_CAMERA)
        fps = int(round(1.0 / (env.dt * skip)))
        imageio.mimwrite(video_path, frames, fps=fps, quality=8, macro_block_size=1)
        print(f"  demo video ({len(frames)/fps:.1f}s @ {fps} fps): {video_path}")

        trace_path = os.path.join(config.VIDEO_DIR, f"{args.env}_{tag}.json")
        with open(trace_path, "w") as f:
            json.dump({**results, "dt": env.dt, "segments": segments,
                       "trace": trace}, f)
        print(f"  command-tracking trace: {trace_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Playground PPO policy")
    parser.add_argument("--run", type=str, required=True,
                        help="Run directory, e.g. models/ppo_Go1JoystickFlatTerrain_seed0")
    parser.add_argument("--env", type=str, default=config.ENV_NAME)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint step folder name (default: latest)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Output filename tag (default: step count)")
    parser.add_argument("--episodes", type=int, default=128,
                        help="Parallel random-command episodes to score (0 to skip)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-video", dest="video", action="store_false")
    args = parser.parse_args()
    evaluate(args)
