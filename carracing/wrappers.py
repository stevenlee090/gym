"""
Custom observation wrappers that avoid the cv2/SDL2 dependency conflict.
Uses PIL for resizing instead of opencv.
"""

import numpy as np
from PIL import Image
import gymnasium as gym
from gymnasium import spaces


class GrayScaleObservation(gym.ObservationWrapper):
    """Convert RGB (H, W, 3) observation to grayscale (H, W, 1)."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(*obs_shape, 1), dtype=np.uint8
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        # Standard luminance weights for RGB→gray
        gray = (obs[..., 0] * 0.299 + obs[..., 1] * 0.587 + obs[..., 2] * 0.114)
        return gray.astype(np.uint8)[..., np.newaxis]


class ResizeObservation(gym.ObservationWrapper):
    """Resize (H, W, C) observation to (new_h, new_w, C) using PIL."""

    def __init__(self, env: gym.Env, shape: tuple[int, int]):
        super().__init__(env)
        self.new_shape = shape  # (H, W)
        old_space = self.observation_space
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(*shape, old_space.shape[2]),
            dtype=np.uint8,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        # PIL expects (W, H) for resize, and handles single/multi channel
        c = obs.shape[2]
        if c == 1:
            img = Image.fromarray(obs[..., 0], mode="L")
            resized = img.resize((self.new_shape[1], self.new_shape[0]), Image.BILINEAR)
            return np.array(resized, dtype=np.uint8)[..., np.newaxis]
        else:
            img = Image.fromarray(obs, mode="RGB")
            resized = img.resize((self.new_shape[1], self.new_shape[0]), Image.BILINEAR)
            return np.array(resized, dtype=np.uint8)
