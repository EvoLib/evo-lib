# SPDX-License-Identifier: MIT
"""Helper for evaluating and visualizing Jumper agents."""

from pathlib import Path
from typing import Any

from evoenv.core.checkpoint import EnvCheckpoint
from evoenv.core.env import Action, Observation
from evoenv.core.sensors import RaySensor
from evoenv.core.task import BaseTask
from evoenv.envs.jumper import JumperEnv
from evoenv.envs.jumper_config import JumperTaskConfig
from evoenv.envs.jumper_defaults import DEFAULT_FPS
from evoenv.renderers.pygame_jumper import run_debug_episode

from evolib import Indiv


class JumperController:
    """Map an EvoLib individual to Jumper jump actions."""

    def __init__(self, indiv: Indiv) -> None:
        self.net: Any = indiv.para["brain"]

    def act(self, observation: Observation) -> Action:
        """Return jump action values produced by the controller."""
        output = self.net.calc(observation)

        jump_signal = float(output[0])
        jump_strength = float(output[1])

        return [jump_signal, jump_strength]


class JumperTask(BaseTask[JumperEnv, JumperController]):
    """Evaluate and visualize individuals on the Jumper environment."""

    def __init__(
        self,
        *,
        task_config: JumperTaskConfig,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            max_steps=task_config.env.max_steps,
            seed=seed,
        )
        self.task_config = task_config
        self.env_config = task_config.env
        self.reward_config = task_config.reward
        self.sensor_config = task_config.sensor

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        seed: int | None = None,
    ) -> "JumperTask":
        """Create a task from a YAML task configuration file."""
        return cls(
            task_config=JumperTaskConfig.from_yaml(path),
            seed=seed,
        )

    def make_sensor(self) -> RaySensor:
        """Create the fixed Jumper ray sensor from the task configuration."""
        return RaySensor(
            length=self.sensor_config.length,
            angle=self.sensor_config.angle,
        )

    @classmethod
    def from_checkpoint(cls, checkpoint: EnvCheckpoint) -> "JumperTask":
        """Create a task from checkpoint metadata."""
        if checkpoint.env.name != "jumper":
            raise ValueError(
                f"Expected 'jumper' checkpoint, got {checkpoint.env.name!r}."
            )

        raw_task_config = checkpoint.env.params.get("task_config")
        if raw_task_config is None:
            raise ValueError("Jumper checkpoint does not contain task_config.")

        return cls(
            task_config=JumperTaskConfig.model_validate(raw_task_config),
            seed=checkpoint.seed,
        )

    def make_env(self) -> JumperEnv:
        """Create a fresh Jumper environment instance."""
        return JumperEnv(
            width=self.env_config.width,
            height=self.env_config.height,
            max_steps=self.env_config.max_steps,
            gravity=self.env_config.gravity,
            jump_velocity=self.env_config.jump_velocity,
            obstacle_speed=self.env_config.obstacle_speed,
            obstacle_width=self.env_config.obstacle_width,
            min_obstacle_height=self.env_config.min_obstacle_height,
            max_obstacle_height=self.env_config.max_obstacle_height,
            min_spawn_gap=self.env_config.min_spawn_gap,
            max_spawn_gap=self.env_config.max_spawn_gap,
            terminate_on_collision=self.env_config.terminate_on_collision,
            collision_penalty=self.reward_config.collision_penalty,
            pass_reward=self.reward_config.pass_reward,
            alive_reward=self.reward_config.alive_reward,
            jump_strength_penalty=self.reward_config.jump_strength_penalty,
            sensor=self.make_sensor(),
        )

    def make_controller(self, indiv: Indiv) -> JumperController:
        """Create the default Jumper controller for one individual."""
        return JumperController(indiv)

    def visualize(
        self,
        indiv: Indiv,
        *,
        steps: int | None = None,
        title: str | None = None,
        filename: str | Path | None = None,
        gif_fps: int = DEFAULT_FPS,
        frame_skip: int = 1,
    ) -> Path | None:
        """Render one debug episode for an individual."""
        display_title = title or "Jumper Debug"
        episode_steps = self.max_steps if steps is None else steps

        return run_debug_episode(
            self.make_env(),
            self.make_controller(indiv),
            steps=episode_steps,
            seed=self.seed,
            title=display_title,
            filename=filename,
            gif_fps=gif_fps,
            frame_skip=frame_skip,
        )
