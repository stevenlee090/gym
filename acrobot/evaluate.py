"""
Evaluate a trained PPO agent on Acrobot-v1 and optionally record a video.

Usage:
    python evaluate.py --model models/ppo_acrobot_seed42/best_model
    python evaluate.py --model models/ppo_acrobot_seed42/best_model --render
    python evaluate.py --model models/ppo_acrobot_seed42/best_model --record
    python evaluate.py --model models/ppo_acrobot_seed42/best_model --episodes 50
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import config

SOLVED_THRESHOLD = -100  # Gymnasium reward threshold for Acrobot-v1


def make_single_env(render_mode=None):
    kwargs = {}
    if render_mode:
        kwargs["render_mode"] = render_mode
    return gym.make(config.ENV_ID, **kwargs)


def run_episodes(model, env, n_episodes):
    rewards = []
    episode_lengths = []
    obs = env.reset()
    ep_reward = 0.0
    ep_length = 0
    completed = 0

    while completed < n_episodes:
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

    return rewards, episode_lengths


def evaluate(args):
    print(f"\nLoading model: {args.model}")
    model = PPO.load(args.model)

    if args.record:
        import imageio
        os.makedirs(config.VIDEO_DIR, exist_ok=True)
        video_path = os.path.join(config.VIDEO_DIR, "acrobot_eval.mp4")

        env = DummyVecEnv([lambda: make_single_env(render_mode="rgb_array")])
        print(f"Running {args.episodes} episode(s) and recording to {video_path}...\n")

        frames = []
        obs = env.reset()
        ep_reward = 0.0
        ep_length = 0
        completed = 0
        rewards = []

        while completed < args.episodes:
            frames.append(env.render())
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward[0]
            ep_length += 1

            if done[0]:
                rewards.append(ep_reward)
                completed += 1
                print(f"  Episode {completed:3d}: reward={ep_reward:8.2f}, length={ep_length}")
                ep_reward = 0.0
                ep_length = 0
                obs = env.reset()

        env.close()
        imageio.mimwrite(video_path, frames, fps=50)
        print(f"\nVideo saved to: {video_path}")

    else:
        render_mode = "human" if args.render else None
        env = DummyVecEnv([lambda: make_single_env(render_mode=render_mode)])
        print(f"Running {args.episodes} evaluation episodes...\n")
        rewards, _ = run_episodes(model, env, args.episodes)
        env.close()

    mean_r = np.mean(rewards)
    std_r = np.std(rewards)
    solved = sum(r >= SOLVED_THRESHOLD for r in rewards)
    print(f"\n{'='*40}")
    print(f"  Episodes:             {args.episodes}")
    print(f"  Mean reward:          {mean_r:.2f} ± {std_r:.2f}")
    print(f"  Min / Max:            {np.min(rewards):.2f} / {np.max(rewards):.2f}")
    print(f"  Solved (≥{SOLVED_THRESHOLD}): {solved}/{args.episodes}")
    print(f"{'='*40}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained PPO on Acrobot-v1")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to saved model (without .zip)")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true",
                        help="Watch the agent live in a pygame window")
    parser.add_argument("--record", action="store_true",
                        help="Record a video to videos/acrobot_eval.mp4")
    args = parser.parse_args()
    evaluate(args)
