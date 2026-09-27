"""
Train a PPO agent on a MuJoCo Playground locomotion task, fully on the GPU.

Physics (MJX) and the policy both run on the GPU with thousands of parallel envs,
so there is no CPU<->GPU transfer in the training loop.

Usage:
    python train.py
    python train.py --timesteps 200000000
    python train.py --env Go1JoystickRoughTerrain --no-dr
"""

import argparse
import functools
import json
import os
import time
import warnings
warnings.filterwarnings("ignore")

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

import jax
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import registry, wrapper
from mujoco_playground.config import locomotion_params
from tensorboardX import SummaryWriter

import config


def build_ppo_params(env_name: str, timesteps: int | None):
    ppo_params = locomotion_params.brax_ppo_config(env_name)
    for key, value in config.OVERRIDES.items():
        ppo_params[key] = value
    if timesteps is not None:
        ppo_params.num_timesteps = timesteps
    return ppo_params


def train(args):
    run_name = f"ppo_{args.env}_seed{args.seed}"
    run_dir = os.path.abspath(os.path.join(config.MODEL_DIR, run_name))
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    env_cfg = registry.get_default_config(args.env)
    env = registry.load(args.env, config=env_cfg)
    eval_env = registry.load(args.env, config=env_cfg)
    ppo_params = build_ppo_params(args.env, args.timesteps)
    randomizer = registry.get_domain_randomizer(args.env) if args.dr else None

    print(f"\n{'='*50}")
    print(f"  {args.env} — Brax PPO (JAX)")
    print(f"  run:     {run_name}")
    print(f"  device:  {jax.devices()[0]}")
    print(f"  steps:   {ppo_params.num_timesteps:,}")
    print(f"  envs:    {ppo_params.num_envs:,}")
    print(f"  DR:      {'on' if randomizer else 'off'}")
    print(f"{'='*50}\n")

    writer = SummaryWriter(os.path.join(config.LOG_DIR, "tensorboard", run_name))
    history = []
    train_history = []
    t_start = time.time()
    t_first = None

    def dump_metrics():
        # Rewritten on every callback so the live dashboard / demo page can read it mid-run.
        summary = {
            "env": args.env,
            "run_name": run_name,
            "device": str(jax.devices()[0]),
            "num_envs": int(ppo_params.num_envs),
            "num_timesteps": int(ppo_params.num_timesteps),
            "domain_randomization": bool(randomizer),
            "elapsed_s": round(time.time() - t_start, 1),
            "jit_time_s": round(t_first or 0.0, 1),
            "history": history,
            "train_history": train_history,
        }
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(summary, f, indent=2)

    def progress(step, metrics):
        if "eval/episode_reward" not in metrics:
            log_training(step, metrics)
        else:
            log_eval(step, metrics)
        dump_metrics()

    def log_training(step, metrics):
        # Called by brax every `training_metrics_steps` with rolling episode stats.
        elapsed = time.time() - t_start
        for key, value in metrics.items():
            writer.add_scalar(key.replace("episode/", "train/"), float(value), step)
        writer.add_scalar("train/elapsed_min", elapsed / 60, step)
        writer.flush()
        reward = float(metrics.get("episode/sum_reward", float("nan")))
        sps = float(metrics.get("episode/sps", 0.0))
        train_history.append({
            "step": int(step),
            "reward": reward,
            "sps": round(sps),
            "wall_time_s": round(elapsed, 1),
        })
        print(f"  step {step:>12,}  train reward {reward:8.2f}  "
              f"[{elapsed/60:5.1f} min, {sps:,.0f} steps/s]", flush=True)

    def log_eval(step, metrics):
        nonlocal t_first
        elapsed = time.time() - t_start
        if step > 0 and t_first is None:
            t_first = elapsed  # first eval after step 0 includes JIT compile time
        reward = float(metrics["eval/episode_reward"])
        reward_std = float(metrics["eval/episode_reward_std"])
        history.append({
            "step": int(step),
            "reward": reward,
            "reward_std": reward_std,
            "wall_time_s": round(elapsed, 1),
        })
        for key, value in metrics.items():
            writer.add_scalar(key, float(value), step)
        writer.flush()
        print(f"  step {step:>12,}  EVAL reward {reward:8.2f} ± {reward_std:6.2f}  "
              f"[{elapsed/60:5.1f} min]", flush=True)

    training_params = dict(ppo_params)
    del training_params["network_factory"]
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **ppo_params.network_factory
    )

    train_fn = functools.partial(
        ppo.train,
        **training_params,
        network_factory=network_factory,
        randomization_fn=randomizer,
        progress_fn=progress,
        seed=args.seed,
        save_checkpoint_path=ckpt_dir,
        log_training_metrics=True,
        training_metrics_steps=args.log_every,
    )
    _, params, _ = train_fn(
        environment=env,
        eval_env=eval_env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )

    total = time.time() - t_start
    dump_metrics()
    writer.close()

    print(f"\nTraining complete in {total/60:.1f} min.")
    print(f"Checkpoints: {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Brax PPO on a Playground task")
    parser.add_argument("--env", type=str, default=config.ENV_NAME)
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override total env steps")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--log-every", type=int, default=1_000_000,
                        help="Env steps between live training-metric logs")
    parser.add_argument("--no-dr", dest="dr", action="store_false",
                        help="Disable domain randomization")
    parser.set_defaults(dr=config.DOMAIN_RANDOMIZATION)
    args = parser.parse_args()
    train(args)
