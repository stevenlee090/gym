"""
Evaluate a trained PPO agent on LunarLander-v3 and optionally record a video.

Usage:
    python evaluate.py --model models/ppo_lunarlander_seed42/best_model
    python evaluate.py --model models/ppo_lunarlander_seed42/best_model --record
    python evaluate.py --model models/ppo_lunarlander_seed42/best_model --episodes 50
"""

import argparse
import os

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

import config


def make_single_env(render_mode=None):
    env_kwargs = {**config.ENV_KWARGS}
    if render_mode:
        env_kwargs["render_mode"] = render_mode
    return gym.make(config.ENV_ID, **env_kwargs)


def evaluate(args):
    print(f"\nLoading model: {args.model}")
    model = PPO.load(args.model)

    if args.record:
        os.makedirs(config.VIDEO_DIR, exist_ok=True)
        env = DummyVecEnv([lambda: make_single_env(render_mode="rgb_array")])
        env = VecVideoRecorder(
            env,
            video_folder=config.VIDEO_DIR,
            record_video_trigger=lambda step: step == 0,
            video_length=3000,
            name_prefix="lunarlander_eval",
        )
    else:
        env = DummyVecEnv([lambda: make_single_env()])

    rewards = []
    episode_lengths = []

    print(f"Running {args.episodes} evaluation episodes...\n")
    obs = env.reset()
    ep_reward = 0.0
    ep_length = 0
    completed = 0

    while completed < args.episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        ep_reward += reward[0]
        ep_length += 1

        if done[0]:
            rewards.append(ep_reward)
            episode_lengths.append(ep_length)
            completed += 1
            print(f"  Episode {completed:3d}: reward={ep_reward:8.2f}, length={ep_length}")
            ep_reward = 0.0
            ep_length = 0
            obs = env.reset()

    env.close()

    mean_r = np.mean(rewards)
    std_r = np.std(rewards)
    print(f"\n{'='*40}")
    print(f"  Episodes:    {args.episodes}")
    print(f"  Mean reward: {mean_r:.2f} ± {std_r:.2f}")
    print(f"  Min / Max:   {np.min(rewards):.2f} / {np.max(rewards):.2f}")
    print(f"  Solved (≥200): {sum(r >= 200 for r in rewards)}/{args.episodes}")
    print(f"{'='*40}\n")

    if args.record:
        print(f"Video saved to: {config.VIDEO_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained PPO on LunarLander-v3")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to saved model (without .zip)")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--record", action="store_true",
                        help="Record a video of the agent")
    args = parser.parse_args()
    evaluate(args)
