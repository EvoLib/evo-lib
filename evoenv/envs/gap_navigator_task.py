# SPDX-License-Identifier: MIT
"""Helper for evaluating and visualizing GapNavigator agents."""

from pathlib import Path
from typing import Any

from evoenv.core.checkpoint import EnvCheckpoint
from evoenv.core.env import Action, Observation
from evoenv.core.evaluator import evaluate_episode
from evoenv.core.sensors import RaySensor
from evoenv.core.task import BaseTask
from evoenv.core.utils import clamp, clamp01
from evoenv.envs.gap_navigator import GapNavigatorEnv, SensorLayout
from evoenv.envs.gap_navigator_config import GapNavigatorTaskConfig
from evoenv.envs.gap_navigator_defaults import DEFAULT_FPS
from evoenv.renderers.pygame_gap_navigator import run_debug_episode

from evolib import Indiv


class GapNavigatorController:
    """Map an EvoLib individual to GapNavigator steering actions."""

    def __init__(self, indiv: Indiv) -> None:
        self.net: Any = indiv.para["brain"]

    def act(self, observation: Observation) -> Action:
        """Return a clipped steering action in [-1, 1]."""
        output = self.net.calc(observation)
        steering = clamp(output[0], -1.0, 1.0)
        return [steering]


class GapNavigatorTask(BaseTask[GapNavigatorEnv, GapNavigatorController]):
    """Evaluate and visualize individuals on the GapNavigator environment."""

    def __init__(
        self,
        *,
        task_config: GapNavigatorTaskConfig,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            max_steps=task_config.env.max_steps,
            seed=seed,
        )

        self.task_config = task_config
        self.env_config = task_config.env
        self.reward_config = task_config.reward
        self.fitness_config = task_config.fitness
        self.sensor_config = task_config.sensors

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        seed: int | None = None,
    ) -> "GapNavigatorTask":
        """Create a task from a YAML task configuration file."""
        return cls(
            task_config=GapNavigatorTaskConfig.from_yaml(path),
            seed=seed,
        )

    @classmethod
    def from_checkpoint(cls, checkpoint: EnvCheckpoint) -> "GapNavigatorTask":
        """Create a task from checkpoint metadata."""
        if checkpoint.env.name != "gap_navigator":
            raise ValueError(
                f"Expected 'gap_navigator' checkpoint, got {checkpoint.env.name!r}."
            )

        raw_task_config = checkpoint.env.params.get("task_config")
        if raw_task_config is None:
            raise ValueError("GapNavigator checkpoint does not contain task_config.")

        return cls(
            task_config=GapNavigatorTaskConfig.model_validate(raw_task_config),
            seed=checkpoint.seed,
        )

    def make_env(self, sensors: SensorLayout | None = None) -> GapNavigatorEnv:
        """Create a fresh GapNavigator environment instance."""
        return GapNavigatorEnv(
            width=self.env_config.width,
            height=self.env_config.height,
            max_steps=self.env_config.max_steps,
            max_sensors=self.sensor_config.max_sensors,
            player_y_offset=self.env_config.player_y_offset,
            player_speed=self.env_config.player_speed,
            row_speed=self.env_config.row_speed,
            row_interval=self.env_config.row_interval,
            obstacle_height=self.env_config.obstacle_height,
            min_gap_width=self.env_config.min_gap_width,
            max_gap_width=self.env_config.max_gap_width,
            edge_margin=self.env_config.edge_margin,
            terminate_on_collision=self.env_config.terminate_on_collision,
            pass_reward=self.reward_config.pass_reward,
            gap_alignment_reward=self.reward_config.gap_alignment_reward,
            movement_penalty=self.reward_config.movement_penalty,
            collision_penalty=self.reward_config.collision_penalty,
            near_wall_penalty=self.reward_config.near_wall_penalty,
            sensors=sensors,
        )

    def make_controller(self, indiv: Indiv) -> GapNavigatorController:
        """Create the default GapNavigator controller for one individual."""
        return GapNavigatorController(indiv)

    def make_sensor_layout(self, indiv: Indiv) -> SensorLayout:
        """
        Build a sensor layout from the evolved vector module.

        Expected vector layout:
        - even indices: sensor lengths in [0, 1]
        - odd indices: sensor angles in [0, 1]

        Inactive sensors are represented by zero-length sensors.
        This preserves stable sensor slots and therefore a stable observation size.
        """
        max_sensors = self.sensor_config.max_sensors
        expected_size = max_sensors * 2
        vector = list(indiv.para["sensors"].vector)

        if len(vector) < expected_size:
            raise ValueError(
                "Sensor module 'sensors' must contain at least "
                f"{expected_size} values, got {len(vector)}."
            )

        sensors: list[RaySensor] = []

        angle_range = self.sensor_config.max_angle - self.sensor_config.min_angle

        for idx in range(0, expected_size, 2):
            length_raw = clamp01(vector[idx])
            angle_raw = clamp01(vector[idx + 1])

            length = length_raw * self.sensor_config.max_length

            if length < self.sensor_config.min_active_length:
                sensors.append(RaySensor(length=0.0, angle=0.0))
                continue

            angle = self.sensor_config.min_angle + angle_raw * angle_range

            sensors.append(
                RaySensor(
                    length=length,
                    angle=angle,
                )
            )

        return tuple(sensors)

    def evaluate(self, indiv: Indiv) -> float:
        """Evaluate an individual with its evolved sensor layout."""
        env = self.make_env(sensors=self.make_sensor_layout(indiv))
        controller = self.make_controller(indiv)

        return evaluate_episode(
            env=env,
            controller=controller,
            seed=self.seed,
            max_steps=self.max_steps,
        )

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
        display_title = title or "GapNavigator Debug"
        sensor_layout = self.make_sensor_layout(indiv)
        episode_steps = self.max_steps if steps is None else steps

        return run_debug_episode(
            self.make_env(sensors=sensor_layout),
            self.make_controller(indiv),
            steps=episode_steps,
            seed=self.seed,
            title=display_title,
            filename=filename,
            gif_fps=gif_fps,
            frame_skip=frame_skip,
        )
