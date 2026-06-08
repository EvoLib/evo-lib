# SPDX-License-Identifier: MIT
"""Helper for evaluating and visualizing GapNavigator agents."""

from pathlib import Path
from typing import Any

from evoenv.core.checkpoint import EnvCheckpoint
from evoenv.core.difficulty import Difficulty
from evoenv.core.env import Action, Observation, validate_action, validate_observation
from evoenv.core.sensors import RaySensor
from evoenv.core.task import BaseTask
from evoenv.core.task_registry import register_task_loader
from evoenv.core.utils import clamp01
from evoenv.envs.gap_navigator import GapNavigatorEnv, SensorLayout
from evoenv.envs.gap_navigator_config import GapNavigatorTaskConfig
from evoenv.envs.gap_navigator_defaults import (
    DEFAULT_DEBUG_EVERY_N_GENERATIONS,
    DEFAULT_DEBUG_MAX_STEPS,
    DEFAULT_FPS,
)
from evoenv.renderers.pygame_gap_navigator import run_debug_episode

from evolib import Indiv


class GapNavigatorController:
    """Map an EvoLib individual to GapNavigator steering actions."""

    def __init__(self, indiv: Indiv, *, module: str = "brain") -> None:
        self.net: Any = indiv.para[module]

    def act(self, observation: Observation) -> Action:
        """Return a clipped steering action in [-1, 1]."""
        output = self.net.calc(observation)
        steering = clamp01(output[0])
        return [steering]


class GapNavigatorTask(BaseTask[GapNavigatorEnv, GapNavigatorController]):
    """Evaluate and visualize individuals on the GapNavigator environment."""

    def __init__(
        self,
        *,
        task_config: GapNavigatorTaskConfig,
        seed: int | None = None,
        module: str = "brain",
        sensor_module: str = "sensors",
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
        self.sensor_module = sensor_module

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
        module: str = "brain",
        sensor_module: str = "sensors",
    ) -> "GapNavigatorTask":
        """Create a task from a YAML task configuration file."""
        return cls(
            task_config=GapNavigatorTaskConfig.from_yaml(path),
            seed=seed,
            module=module,
            sensor_module=sensor_module,
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
            sensors=sensors,
        )

    def make_controller(self, indiv: Indiv) -> GapNavigatorController:
        """Create the default GapNavigator controller for one individual."""
        return GapNavigatorController(indiv, module=self.module)

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
        vector = list(indiv.para[self.sensor_module].vector)

        if len(vector) < expected_size:
            raise ValueError(
                f"Sensor module {self.sensor_module!r} must contain at least "
                f"{expected_size} values, got {len(vector)}."
            )

        sensors: list[RaySensor] = []

        angle_range = self.sensor_config.max_angle - self.sensor_config.min_angle

        for idx in range(0, expected_size, 2):
            length_raw = self._clamp01(vector[idx])
            angle_raw = self._clamp01(vector[idx + 1])

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

    def compute_reward(self, info: dict[str, Any]) -> float:
        """Compute task reward from GapNavigator simulation info."""
        reward = 0.0

        reward += float(info["gap_alignment"]) * self.reward_config.gap_alignment_reward

        reward -= abs(float(info["steering"])) * self.reward_config.movement_penalty

        if bool(info["has_collision"]):
            reward -= self.reward_config.collision_penalty

        if bool(info["near_wall"]):
            reward -= self.reward_config.near_wall_penalty

        if bool(info["row_passed"]) and self.env_config.terminate_on_collision:
            reward += self.reward_config.pass_reward

        return reward

    def evaluate(self, indiv: Indiv) -> float:
        """Run one full episode and return the accumulated task reward."""
        sensor_layout = self.make_sensor_layout(indiv)
        env = self.make_env(sensors=sensor_layout)
        controller = self.make_controller(indiv)

        observation = env.reset(seed=self.seed)
        validate_observation(env, observation)

        total_reward = 0.0

        for _ in range(self.max_steps):
            action = controller.act(observation)
            validate_action(env, action)

            observation, _env_reward, done, info = env.step(action)
            validate_observation(env, observation)

            total_reward += self.compute_reward(info)

            if done:
                break

        return total_reward

    def visualize(
        self,
        indiv: Indiv,
        *,
        generation: int,
        every: int = DEFAULT_DEBUG_EVERY_N_GENERATIONS,
        steps: int = DEFAULT_DEBUG_MAX_STEPS,
        title: str | None = None,
        filename: str | Path | None = None,
        gif_fps: int = DEFAULT_FPS,
        frame_skip: int = 1,
    ) -> Path | None:
        """Render one debug episode for an individual."""
        display_title = title or f"GapNavigator Training Debug - Gen {generation}"
        sensor_layout = self.make_sensor_layout(indiv)
        episode_steps = self.max_steps if steps is None else steps

        return run_debug_episode(
            self.make_env(sensors=sensor_layout),
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
            reward_fn=self.compute_reward,
        )


def load_gap_navigator_task(checkpoint: EnvCheckpoint) -> GapNavigatorTask:
    """Create a GapNavigator task from checkpoint metadata."""
    raw_task_config = checkpoint.env.params.get("task_config")
    if raw_task_config is None:
        raise ValueError("GapNavigator checkpoint does not contain task_config.")

    return GapNavigatorTask(
        task_config=GapNavigatorTaskConfig.model_validate(raw_task_config),
        seed=checkpoint.seed,
        difficulty=checkpoint.env.difficulty or Difficulty.MEDIUM,
    )


def register_gap_navigator_task() -> None:
    """Register the GapNavigator task loader."""
    register_task_loader("gap_navigator", load_gap_navigator_task)
