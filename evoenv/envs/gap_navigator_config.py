# SPDX-License-Identifier: MIT
"""Pydantic configuration models for the GapNavigator example."""

from __future__ import annotations

from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class GapNavigatorEnvConfig(BaseModel):
    """Simulation parameters for the GapNavigator environment."""

    model_config = ConfigDict(extra="forbid")

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    max_steps: int = Field(gt=0)

    player_y_offset: int = Field(ge=0)
    player_speed: float = Field(gt=0.0)

    row_speed: float = Field(gt=0.0)
    row_interval: int = Field(gt=0)
    obstacle_height: int = Field(gt=0)
    min_gap_width: float = Field(gt=0.0)
    max_gap_width: float = Field(gt=0.0)
    edge_margin: float = Field(ge=0.0)
    terminate_on_collision: bool

    @model_validator(mode="after")
    def validate_geometry(self) -> Self:
        """Validate geometric constraints that depend on multiple fields."""
        if self.min_gap_width > self.max_gap_width:
            raise ValueError(
                "min_gap_width must be less than or equal to max_gap_width."
            )

        if self.max_gap_width + 2.0 * self.edge_margin > self.width:
            raise ValueError(
                "max_gap_width plus both edge margins must fit within width."
            )

        if self.player_y_offset >= self.height:
            raise ValueError("player_y_offset must be smaller than height.")

        return self


class GapNavigatorRewardConfig(BaseModel):
    """Reward shaping parameters for the GapNavigator task."""

    model_config = ConfigDict(extra="forbid")

    pass_reward: float
    gap_alignment_reward: float = Field(ge=0.0)
    movement_penalty: float = Field(ge=0.0)
    collision_penalty: float = Field(ge=0.0)
    near_wall_penalty: float = Field(ge=0.0)


class GapNavigatorFitnessConfig(BaseModel):
    """Fitness parameters for the GapNavigator task."""

    model_config = ConfigDict(extra="forbid")

    sensor_count_penalty: float = Field(ge=0.0)
    sensor_length_penalty: float = Field(ge=0.0)
    sensor_length_scale: float = Field(gt=0.0)


class GapNavigatorSensorConfig(BaseModel):
    """Encoding parameters for evolved GapNavigator sensors."""

    model_config = ConfigDict(extra="forbid")

    max_sensors: int = Field(gt=0)
    max_length: float = Field(gt=0.0)
    min_active_length: float = Field(ge=0.0)
    min_angle: float
    max_angle: float

    @model_validator(mode="after")
    def validate_sensor_range(self) -> Self:
        """Validate sensor length and angle ranges."""
        if self.min_active_length > self.max_length:
            raise ValueError("min_active_length must not exceed max_length.")

        if self.min_angle >= self.max_angle:
            raise ValueError("min_angle must be smaller than max_angle.")

        return self


class GapNavigatorTaskConfig(BaseModel):
    """Complete YAML configuration for one GapNavigator experiment."""

    model_config = ConfigDict(extra="forbid")

    env: GapNavigatorEnvConfig
    reward: GapNavigatorRewardConfig
    fitness: GapNavigatorFitnessConfig
    sensors: GapNavigatorSensorConfig

    def to_yaml_dict(self) -> dict[str, object]:
        """Return a YAML-serializable representation of the configuration."""
        return self.model_dump(mode="json")

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        """Load and validate a GapNavigator task configuration from YAML."""
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as file:
            raw_config = yaml.safe_load(file) or {}

        return cls.model_validate(raw_config)
