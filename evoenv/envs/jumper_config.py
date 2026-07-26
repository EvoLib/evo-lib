# SPDX-License-Identifier: MIT
"""Pydantic configuration models for the Jumper example."""

from __future__ import annotations

from typing import Self

from evoenv.core.config import StrictConfigModel, YamlConfigModel
from pydantic import Field, model_validator


class JumperEnvConfig(StrictConfigModel):
    """Simulation parameters for the Jumper environment."""

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    max_steps: int = Field(gt=0)

    gravity: float = Field(gt=0.0)
    jump_velocity: float = Field(gt=0.0)

    obstacle_speed: float = Field(gt=0.0)
    obstacle_width: int = Field(gt=0)
    min_obstacle_height: int = Field(gt=0)
    max_obstacle_height: int = Field(gt=0)
    min_spawn_gap: int = Field(ge=0)
    max_spawn_gap: int = Field(ge=0)

    terminate_on_collision: bool

    @model_validator(mode="after")
    def validate_ranges(self) -> Self:
        """Validate parameter ranges that depend on multiple fields."""
        if self.min_obstacle_height > self.max_obstacle_height:
            raise ValueError(
                "min_obstacle_height must be less than or equal to "
                "max_obstacle_height."
            )

        if self.min_spawn_gap > self.max_spawn_gap:
            raise ValueError(
                "min_spawn_gap must be less than or equal to max_spawn_gap."
            )

        return self


class JumperRewardConfig(StrictConfigModel):
    """Reward parameters for the Jumper task."""

    collision_penalty: float = Field(ge=0.0)
    pass_reward: float
    alive_reward: float
    jump_strength_penalty: float = Field(ge=0.0)


class JumperSensorConfig(StrictConfigModel):
    """Sensor parameters for the fixed Jumper ray sensor."""

    length: float = Field(gt=0.0)
    angle: float


class JumperTaskConfig(YamlConfigModel):
    """Complete YAML configuration for one Jumper experiment."""

    env: JumperEnvConfig
    reward: JumperRewardConfig
    sensor: JumperSensorConfig
