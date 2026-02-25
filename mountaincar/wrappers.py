"""
Reward shaping wrapper for MountainCarContinuous-v0.

Kinetic energy shaping:  r' = r_env + w * velocity²

The car is rewarded for having kinetic energy (speed in either direction).
This immediately incentivises the rocking behaviour needed to build momentum,
without changing the terminal goal (the original +100 bonus is unchanged).

Unlike potential-based shaping, this is NOT theoretically neutral with respect
to the optimal policy — it slightly prefers faster solutions — but in practice
the shaped and unshaped optimal policies both reach the goal as quickly as
possible, so the difference is negligible.
"""

import gymnasium as gym
import numpy as np


class KineticEnergyShapingWrapper(gym.Wrapper):
    """Adds w * velocity² to every step reward."""

    def __init__(self, env: gym.Env, weight: float = 0.1):
        super().__init__(env)
        self.weight = weight

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        velocity = float(obs[1])
        return obs, reward + self.weight * velocity ** 2, terminated, truncated, info
