"""
Evaluate a trained SAC agent on CarRacing-v3 and optionally record a video.

Usage:
    python evaluate.py --model models/sac_carracing_seed42/best_model
    python evaluate.py --model models/sac_carracing_seed42/best_model --render
    python evaluate.py --model models/sac_carracing_seed42/best_model --record
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from wrappers import GrayScaleObservation, ResizeObservation
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

import config


def wrap_obs(env: gym.Env) -> gym.Env:
    if config.OBS_GRAYSCALE:
        env = GrayScaleObservation(env)
    if config.OBS_RESIZE:
        env = ResizeObservation(env, shape=config.OBS_RESIZE)
    return env


def make_single_env(render_mode=None):
    env_kwargs = {**config.ENV_KWARGS}
    if render_mode:
        env_kwargs["render_mode"] = render_mode
    env = DummyVecEnv([lambda: wrap_obs(gym.make(config.ENV_ID, **env_kwargs))])
    env = VecFrameStack(env, n_stack=config.N_STACK)
    return env


def evaluate(args):
    print(f"\nLoading model: {args.model}")
    model = PPO.load(args.model)

    if args.record:
        import imageio
        os.makedirs(config.VIDEO_DIR, exist_ok=True)
        video_path = os.path.join(config.VIDEO_DIR, "carracing_eval.mp4")

        env = make_single_env(render_mode="rgb_array")
        print(f"Running {args.episodes} episode(s) and recording to {video_path}...\n")

        frames = []
        rewards = []
        obs = env.reset()
        ep_reward = 0.0
        ep_length = 0
        completed = 0

        while completed < args.episodes:
            # Render the base env (not stacked) for a clean video frame
            frame = env.envs[0].env.render()
            frames.append(frame)
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
        env = make_single_env(render_mode=render_mode)
        print(f"Running {args.episodes} evaluation episodes...\n")

        rewards = []
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
                completed += 1
                print(f"  Episode {completed:3d}: reward={ep_reward:8.2f}, length={ep_length}")
                ep_reward = 0.0
                ep_length = 0
                obs = env.reset()

        env.close()

    mean_r = np.mean(rewards)
    std_r = np.std(rewards)
    print(f"\n{'='*40}")
    print(f"  Episodes:      {args.episodes}")
    print(f"  Mean reward:   {mean_r:.2f} ± {std_r:.2f}")
    print(f"  Min / Max:     {np.min(rewards):.2f} / {np.max(rewards):.2f}")
    print(f"  Solved (≥900): {sum(r >= 900 for r in rewards)}/{args.episodes}")
    print(f"{'='*40}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained SAC on CarRacing-v3")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to saved model (without .zip)")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true",
                        help="Watch the agent live in a pygame window")
    parser.add_argument("--record", action="store_true",
                        help="Record a video to videos/carracing_eval.mp4")
    args = parser.parse_args()
    evaluate(args)
