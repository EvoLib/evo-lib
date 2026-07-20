# SPDX-License-Identifier: MIT
"""Helper for evaluating and visualizing LineFollower agents."""

from pathlib import Path
from typing import Any

from evoenv.core.checkpoint import EnvCheckpoint
from evoenv.core.difficulty import Difficulty
from evoenv.core.env import Action, Observation
from evoenv.core.task import BaseTask
from evoenv.core.task_registry import register_task_loader
from evoenv.core.utils import clamp
from evoenv.envs.line_follower import LineFollowerEnv
from evoenv.envs.line_follower_config import LineFollowerTaskConfig
from evoenv.envs.line_follower_defaults import (
    DEFAULT_DEBUG_EVERY_N_GENERATIONS,
    DEFAULT_FPS,
)
from evoenv.renderers.pygame_line_follower import run_debug_episode

from evolib import Indiv


class LineFollowerController:
    """Map an EvoLib individual to LineFollower steering actions."""

    def __init__(self, indiv: Indiv, *, module: str = "brain") -> None:
        self.net: Any = indiv.para[module]

    def act(self, observation: Observation) -> Action:
        """Return a clipped steering action in [-1, 1]."""
        output = self.net.calc(observation)
        turn = clamp(output[0], -1.0, 1.0)
        return [turn]


class LineFollowerTask(BaseTask[LineFollowerEnv, LineFollowerController]):
    """Evaluate and visualize individuals on the LineFollower environment."""

    def __init__(
        self,
        *,
        task_config: LineFollowerTaskConfig,
        seed: int | None = None,
        module: str = "brain",
        difficulty: str | Difficulty = Difficulty.MEDIUM,
    ) -> None:
        super().__init__(
            width=task_config.env.width,
            height=task_config.env.height,
            max_steps=task_config.env.max_steps,
            seed=seed,
            module=module,
            difficulty=difficulty,
        )
        self.task_config = task_config
        self.env_config = task_config.env
        self.reward_config = task_config.reward

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        seed: int | None = None,
        module: str = "brain",
        difficulty: str | Difficulty = Difficulty.MEDIUM,
    ) -> "LineFollowerTask":
        """Create a task from a YAML task configuration file."""
        return cls(
            task_config=LineFollowerTaskConfig.from_yaml(path),
            seed=seed,
            module=module,
            difficulty=difficulty,
        )

    def make_env(self) -> LineFollowerEnv:
        """Create a fresh LineFollower environment instance."""
        return LineFollowerEnv(
            width=self.env_config.width,
            height=self.env_config.height,
            max_steps=self.env_config.max_steps,
            line_complexity=self.env_config.line_complexity,
            line_width=self.env_config.line_width,
            base_speed=self.env_config.base_speed,
            turn_strength=self.env_config.turn_strength,
            max_missed_line_steps=self.env_config.max_missed_line_steps,
            progress_reward_scale=self.reward_config.progress_reward_scale,
            missed_line_penalty=self.reward_config.missed_line_penalty,
        )

    def make_controller(self, indiv: Indiv) -> LineFollowerController:
        """Create the default LineFollower controller for one individual."""
        return LineFollowerController(indiv, module=self.module)

    def visualize(
        self,
        indiv: Indiv,
        *,
        generation: int,
        every: int = DEFAULT_DEBUG_EVERY_N_GENERATIONS,
        steps: int | None = None,
        title: str | None = None,
        filename: str | Path | None = None,
        gif_fps: int = DEFAULT_FPS,
        frame_skip: int = 1,
    ) -> Path | None:
        """Render one debug episode for an individual."""
        display_title = title or f"Training Debug - Gen {generation}"
        episode_steps = self.max_steps if steps is None else steps

        return run_debug_episode(
            self.make_env(),
            self.make_controller(indiv),
            enabled=True,
            generation=generation,
            every=every,
            steps=episode_steps,
            seed=self.seed,
            title=display_title,
            filename=filename,
            gif_fps=gif_fps,
            frame_skip=frame_skip,
        )


def load_line_follower_task(checkpoint: EnvCheckpoint) -> LineFollowerTask:
    """Create a LineFollower task from checkpoint metadata."""
    raw_task_config = checkpoint.env.params.get("task_config")
    if raw_task_config is None:
        raise ValueError("LineFollower checkpoint does not contain task_config.")

    return LineFollowerTask(
        task_config=LineFollowerTaskConfig.model_validate(raw_task_config),
        seed=checkpoint.seed,
        difficulty=checkpoint.env.difficulty or Difficulty.MEDIUM,
    )


def register_line_follower_task() -> None:
    """Register the LineFollower task loader."""
    register_task_loader("line_follower", load_line_follower_task)
