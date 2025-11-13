# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union, cast

try:
    import gymnasium as gym
except ImportError as e:
    raise ImportError(
        "Gymnasium is required for GymEnv. Please install with `pip install gymnasium`."
    ) from e

try:
    import imageio
except ImportError as e:
    raise ImportError(
        "imageio is required for GymEnv visualization. "
        "Please install with `pip install imageio`."
    ) from e

import numpy as np

if TYPE_CHECKING:
    from evolib import Individual


class GymEnv:
    """Thin wrapper to run OpenAI Gymnasium environments with EvoLib Individuals."""

    def __init__(
        self,
        env_name: str,
        max_steps: int = 500,
        deterministic_init: bool = False,
        **env_kwargs: Any,
    ):
        """
        Initialize a Gym environment.

        Args:
            env_name: Name of the Gymnasium environment, e.g. "FrozenLake-v1".
            max_steps: Maximum number of steps per episode.
            **env_kwargs: Extra arguments passed directly to gym.make(),
                          e.g. is_slippery=False for FrozenLake.
        """
        self.env_name = env_name
        self.max_steps = max_steps
        self.deterministic_init = deterministic_init
        self.env_kwargs = env_kwargs
        # headless env (no Render)
        self.env = gym.make(env_name, **env_kwargs)

    def evaluate(
        self,
        indiv: Individual,
        module: str = "brain",
        episodes: int = 1,
        seed: int | None = None,
    ) -> float:
        """
        Run one or multiple episodes headless and return average total reward.

        Args:
            indiv: Individual whose network acts in the environment.
            module: Which module in para to use for decision making.
            episodes: How many episodes to average over (default: 1).

        Returns:
            Average total reward across all episodes.
        """
        total_reward = 0.0

        for _ in range(episodes):
            obs, _ = self.env.reset(seed=seed)

            if self.deterministic_init:
                unwrapped = self.env.unwrapped
                if hasattr(unwrapped, "state") and isinstance(
                    unwrapped.state, np.ndarray
                ):
                    unwrapped.state = np.zeros_like(unwrapped.state)
                    obs = unwrapped.state
                else:
                    obs = np.zeros_like(obs)

                assert np.allclose(
                    obs, 0.0
                ), "Deterministic init failed: state not zeroed."

            net = indiv.para[module].net
            net.reset(full=True)  # Reset recurrent/internal state

            ep_reward = 0.0

            for _ in range(self.max_steps):
                if isinstance(obs, np.ndarray):
                    obs_list = obs.tolist()
                elif np.isscalar(obs):
                    obs_list = [float(cast(float, obs))]
                else:
                    obs_list = list(obs)

                action = net.calc(obs_list)

                # Discrete Action-Spaces --> argmax
                if hasattr(self.env.action_space, "n"):
                    action = int(np.argmax(action))
                else:
                    action = np.array(action, dtype=np.float32)

                obs, reward, terminated, truncated, _ = self.env.step(action)

                ep_reward += float(reward)

                if terminated or truncated:
                    break

            total_reward += ep_reward

        return total_reward / episodes

    def visualize(
        self,
        indiv: Individual,
        gen: int,
        filename: str | None = None,
        fps: int = 30,
        module: str = "brain",
        seed: int | None = None,
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
            self.env_name,
            render_mode="rgb_array",
            max_episode_steps=self.max_steps,
            **self.env_kwargs,
        )
        obs, _ = env.reset(seed=seed)

        if self.deterministic_init:
            unwrapped = env.unwrapped
            if hasattr(unwrapped, "state") and isinstance(unwrapped.state, np.ndarray):
                unwrapped.state = np.zeros_like(unwrapped.state)
                obs = unwrapped.state
            else:
                obs = np.zeros_like(obs)

            assert np.allclose(obs, 0.0), "Deterministic init failed: state not zeroed."

        net = indiv.para[module].net
        net.reset(full=True)  # Reset recurrent/internal state

        RenderFrame = Union[np.ndarray, list[np.ndarray], None]
        frames: list[np.ndarray] = []

        for _ in range(self.max_steps):
            if isinstance(obs, np.ndarray):
                obs_list = obs.tolist()
            elif np.isscalar(obs):
                obs_list = [float(cast(float, obs))]
            else:
                obs_list = list(obs)

            action = net.calc(obs_list)

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

        imageio.mimsave(filename, cast(list[Any], frames), fps=fps)

        return filename
