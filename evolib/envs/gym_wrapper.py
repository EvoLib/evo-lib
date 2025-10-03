# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Union

import gymnasium as gym
import imageio
import numpy as np

from evolib import Individual  # dein Core-Typ


class GymEnvWrapper:
    """Thin wrapper to run OpenAI Gymnasium environments with EvoLib Individuals."""

    def __init__(self, env_name: str, max_steps: int = 500):
        self.env_name = env_name
        self.max_steps = max_steps
        # headless env (no Render)
        self.env = gym.make(env_name)

    def evaluate(self, indiv: Individual, module: str = "brain") -> float:
        """Run one episode headless and return total reward (fitness)."""
        obs, _ = self.env.reset()
        total_reward = 0.0

        for _ in range(self.max_steps):
            obs_list = obs.tolist() if isinstance(obs, np.ndarray) else list(obs)
            action = indiv.para[module].net.calc(obs_list)

            # Discrete Action-Spaces --> argmax
            if hasattr(self.env.action_space, "n"):
                action = int(np.argmax(action))
            else:
                action = np.array(action, dtype=np.float32)

            obs, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break

        return total_reward

    def visualize(
        self,
        indiv: Individual,
        gen: int,
        filename: str | None = None,
        fps: int = 30,
        module: str = "brain",
    ) -> str:
        """
        Render an episode with the given individual and save as GIF using imageio.

        Args:
            indiv: Individual to visualize.
            gen: Generation number (used in default filename).
            filename: Optional filename for output GIF.
            fps: Frames per second for GIF.
            module: Which module in para to use for decision making.

        Returns:
            Path to saved GIF.
        """
        env = gym.make(
            self.env_name, render_mode="rgb_array", max_episode_steps=self.max_steps
        )
        obs, _ = env.reset()

        RenderFrame = Union[np.ndarray, list[np.ndarray], None]
        frames: list[np.ndarray] = []

        for _ in range(self.max_steps):
            obs_list = obs.tolist() if isinstance(obs, np.ndarray) else list(obs)
            action = indiv.para[module].net.calc(obs_list)

            if hasattr(env.action_space, "n"):
                action = int(np.argmax(action))
            else:
                action = np.array(action, dtype=np.float32)

            obs, reward, terminated, truncated, _ = env.step(action)

            frame: RenderFrame = env.render()
            if isinstance(frame, np.ndarray):
                frames.append(frame)

            if terminated or truncated:
                break

        env.close()

        if filename is None:
            filename = f"{self.env_name}_gen{gen:04d}.gif"

        imageio.mimsave(filename, frames, fps=fps)  # type: ignore[arg-type]
        return filename
