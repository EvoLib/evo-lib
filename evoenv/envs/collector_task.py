# SPDX-License-Identifier: MIT
"""Helper for evaluating and visualizing Collector agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evoenv.core.checkpoint import EnvCheckpoint
from evoenv.core.env import Action, Observation
from evoenv.core.task import BaseTask
from evoenv.core.task_registry import register_task_loader
from evoenv.core.utils import clamp, clamp01
from evoenv.envs.collector import CollectorEnv
from evoenv.envs.collector_config import CollectorTaskConfig
from evoenv.envs.collector_defaults import (
    DEFAULT_DEBUG_EVERY_N_GENERATIONS,
    DEFAULT_FPS,
)
from evoenv.renderers.pygame_collector import run_debug_episode

from evolib import Indiv


class CollectorController:
    """Map an EvoLib individual to Collector movement actions."""

    def __init__(self, indiv: Indiv, *, module: str = "brain") -> None:
        self.net: Any = indiv.para[module]

    def act(self, observation: Observation) -> Action:
        """Return clipped movement actions."""
        output = self.net.calc(observation)

        turn = clamp(output[0], -1.0, 1.0)
        throttle = clamp01((float(output[1]) + 1.0) * 0.5)

        return [turn, throttle]


class CollectorTask(BaseTask[CollectorEnv, CollectorController]):
    """Evaluate and visualize individuals on the Collector environment."""

    def __init__(
        self,
        *,
        task_config: CollectorTaskConfig,
        seed: int | None = None,
        module: str = "brain",
    ) -> None:
        super().__init__(
            width=task_config.env.width,
            height=task_config.env.height,
            max_steps=task_config.env.max_steps,
            seed=seed,
            module=module,
            difficulty="standard",
        )
        self.task_config = task_config
        self.env_config = task_config.env
        self.reward_config = task_config.reward
        self.sensor_config = task_config.sensor
        self.exploration_config = task_config.exploration

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        seed: int | None = None,
        module: str = "brain",
    ) -> "CollectorTask":
        """Create a task from a YAML task configuration file."""
        return cls(
            task_config=CollectorTaskConfig.from_yaml(path),
            seed=seed,
            module=module,
        )

    def make_env(self) -> CollectorEnv:
        """Create a fresh Collector environment instance."""
        return CollectorEnv(
            width=self.env_config.width,
            height=self.env_config.height,
            max_steps=self.env_config.max_steps,
            agent_radius=self.env_config.agent_radius,
            base_speed=self.env_config.base_speed,
            turn_strength=self.env_config.turn_strength,
            food_count=self.env_config.food_count,
            food_radius=self.env_config.food_radius,
            collect_radius=self.env_config.collect_radius,
            obstacle_count=self.env_config.obstacle_count,
            obstacle_min_size=self.env_config.obstacle_min_size,
            obstacle_max_size=self.env_config.obstacle_max_size,
            spawn_margin=self.env_config.spawn_margin,
            terminate_on_collision=self.env_config.terminate_on_collision,
            food_reward=self.reward_config.food_reward,
            distance_reward=self.reward_config.distance_reward,
            exploration_reward=self.reward_config.exploration_reward,
            collision_penalty=self.reward_config.collision_penalty,
            step_penalty=self.reward_config.step_penalty,
            turn_penalty=self.reward_config.turn_penalty,
            ray_length=self.sensor_config.ray_length,
            ray_angles=self.sensor_config.ray_angles,
            exploration_cell_size=self.exploration_config.cell_size,
        )

    def make_controller(self, indiv: Indiv) -> CollectorController:
        """Create the default Collector controller for one individual."""
        return CollectorController(indiv, module=self.module)

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
        display_title = title or f"Collector Training Debug - Gen {generation}"
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


def load_collector_task(checkpoint: EnvCheckpoint) -> CollectorTask:
    """Create a Collector task from checkpoint metadata."""
    raw_task_config = checkpoint.env.params.get("task_config")

    if raw_task_config is None:
        raise ValueError("Collector checkpoint does not contain task_config.")

    module = checkpoint.env.params.get("module", "brain")
    if not isinstance(module, str):
        raise ValueError("Collector checkpoint module must be a string.")

    return CollectorTask(
        task_config=CollectorTaskConfig.model_validate(raw_task_config),
        seed=checkpoint.seed,
        module=module,
    )


def register_collector_task() -> None:
    """Register the Collector task loader."""
    register_task_loader("collector", load_collector_task)
