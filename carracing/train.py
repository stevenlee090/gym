"""
Train a SAC agent on CarRacing-v3 (continuous action space, pixel observations).

Usage:
    python train.py
    python train.py --timesteps 3000000 --n-envs 4
    python train.py --resume models/sac_carracing_seed42/checkpoints/sac_carracing_500000_steps
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)

import config


def make_env(n_envs: int, seed: int):
    env = make_vec_env(
        config.ENV_ID,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=config.ENV_KWARGS,
    )
    env = VecFrameStack(env, n_stack=config.N_STACK)
    return env


def make_eval_env(seed: int):
    env = make_vec_env(
        config.ENV_ID,
        n_envs=1,
        seed=seed + 1000,
        env_kwargs=config.ENV_KWARGS,
    )
    env = VecFrameStack(env, n_stack=config.N_STACK)
    env = VecTransposeImage(env)  # Match the auto-wrap SB3 applies to train env
    return env


def build_callbacks(eval_env, run_name: str, n_envs: int) -> CallbackList:
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(config.MODEL_DIR, run_name),
        log_path=os.path.join(config.LOG_DIR, run_name),
        eval_freq=max(config.EVAL_FREQ // n_envs, 1),
        n_eval_episodes=config.N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(config.CHECKPOINT_FREQ // n_envs, 1),
        save_path=os.path.join(config.MODEL_DIR, run_name, "checkpoints"),
        name_prefix=config.MODEL_NAME,
        verbose=1,
    )
    return CallbackList([eval_callback, checkpoint_callback])


def train(args):
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    run_name = f"{config.MODEL_NAME}_seed{args.seed}"
    tb_log = os.path.join(config.LOG_DIR, "tensorboard")

    print(f"\n{'='*50}")
    print(f"  CarRacing-v3 SAC Training (continuous)")
    print(f"  run:    {run_name}")
    print(f"  device: cpu")
    print(f"  steps:  {args.timesteps:,}")
    print(f"  envs:   {args.n_envs}  (frame stack: {config.N_STACK})")
    print(f"{'='*50}\n")

    train_env = make_env(n_envs=args.n_envs, seed=args.seed)
    eval_env = make_eval_env(seed=args.seed)

    if args.resume:
        print(f"Resuming from: {args.resume}")
        model = SAC.load(
            args.resume,
            env=train_env,
            device="cpu",
            tensorboard_log=tb_log,
        )
    else:
        model = SAC(
            env=train_env,
            verbose=1,
            seed=args.seed,
            device="cpu",
            tensorboard_log=tb_log,
            **config.SAC_KWARGS,
        )

    callbacks = build_callbacks(eval_env, run_name, args.n_envs)

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        tb_log_name=run_name,
        reset_num_timesteps=not args.resume,
        progress_bar=True,
    )

    final_path = os.path.join(config.MODEL_DIR, run_name, f"{config.MODEL_NAME}_final")
    model.save(final_path)
    print(f"\nTraining complete. Final model saved to: {final_path}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC on CarRacing-v3")
    parser.add_argument("--timesteps", type=int, default=config.TOTAL_TIMESTEPS)
    parser.add_argument("--n-envs", type=int, default=config.N_ENVS)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to model zip to resume from")
    args = parser.parse_args()
    train(args)
