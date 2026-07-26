# SPDX-License-Identifier: MIT
"""Pydantic configuration models for the Jumper example."""

from __future__ import annotations

from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class JumperEnvConfig(BaseModel):
    """Simulation parameters for the Jumper environment."""

    model_config = ConfigDict(extra="forbid")

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


class JumperRewardConfig(BaseModel):
    """Reward parameters for the Jumper task."""

    model_config = ConfigDict(extra="forbid")

    collision_penalty: float = Field(ge=0.0)
    pass_reward: float
    alive_reward: float
    jump_strength_penalty: float = Field(ge=0.0)


class JumperSensorConfig(BaseModel):
    """Sensor parameters for the fixed Jumper ray sensor."""

    model_config = ConfigDict(extra="forbid")

    length: float = Field(gt=0.0)
    angle: float


class JumperTaskConfig(BaseModel):
    """Complete YAML configuration for one Jumper experiment."""

    model_config = ConfigDict(extra="forbid")

    env: JumperEnvConfig
    reward: JumperRewardConfig
    sensor: JumperSensorConfig

    def to_yaml_dict(self) -> dict[str, object]:
        """Return a YAML-serializable representation of the configuration."""
        return self.model_dump(mode="json")

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        """Load and validate a Jumper task configuration from YAML."""
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as file:
            raw_config = yaml.safe_load(file) or {}

        return cls.model_validate(raw_config)
