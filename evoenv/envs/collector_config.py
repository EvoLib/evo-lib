# SPDX-License-Identifier: MIT
"""Pydantic configuration models for the Collector example."""

from __future__ import annotations

from typing import Self

from evoenv.core.config import StrictConfigModel, YamlConfigModel
from pydantic import ConfigDict, Field, model_validator


class CollectorEnvConfig(StrictConfigModel):
    """Simulation parameters for the Collector environment."""

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    max_steps: int = Field(gt=0)

    agent_radius: int = Field(gt=0)
    base_speed: float = Field(gt=0.0)
    turn_strength: float = Field(gt=0.0)

    food_count: int = Field(ge=1)
    food_radius: int = Field(gt=0)
    collect_radius: float = Field(gt=0.0)

    obstacle_count: int = Field(ge=0)
    obstacle_min_size: int = Field(gt=0)
    obstacle_max_size: int = Field(gt=0)
    spawn_margin: int = Field(ge=0)

    terminate_on_collision: bool

    @model_validator(mode="after")
    def validate_ranges(self) -> Self:
        """Validate parameter ranges that depend on multiple fields."""
        if self.obstacle_min_size > self.obstacle_max_size:
            raise ValueError(
                "obstacle_min_size must be less than or equal to obstacle_max_size."
            )

        if self.agent_radius * 2 >= self.width:
            raise ValueError("agent_radius leaves no horizontal movement space.")

        if self.agent_radius * 2 >= self.height:
            raise ValueError("agent_radius leaves no vertical movement space.")

        obstacle_required_width = self.spawn_margin * 2 + self.obstacle_max_size
        obstacle_required_height = self.spawn_margin * 2 + self.obstacle_max_size

        if obstacle_required_width >= self.width:
            raise ValueError(
                "spawn_margin and obstacle_max_size leave no horizontal "
                "obstacle spawn space."
            )

        if obstacle_required_height >= self.height:
            raise ValueError(
                "spawn_margin and obstacle_max_size leave no vertical "
                "obstacle spawn space."
            )

        food_required_width = (self.spawn_margin + self.food_radius) * 2
        food_required_height = (self.spawn_margin + self.food_radius) * 2

        if food_required_width >= self.width:
            raise ValueError(
                "spawn_margin and food_radius leave no horizontal food spawn space."
            )

        if food_required_height >= self.height:
            raise ValueError(
                "spawn_margin and food_radius leave no vertical food spawn space."
            )

        if self.collect_radius < self.food_radius:
            raise ValueError(
                "collect_radius must be greater than or equal to food_radius."
            )

        return self


class CollectorRewardConfig(StrictConfigModel):
    """Reward parameters for the Collector task."""

    food_reward: float
    distance_reward: float
    exploration_reward: float
    collision_penalty: float = Field(ge=0.0)
    step_penalty: float = Field(ge=0.0)
    turn_penalty: float = Field(ge=0.0)


class CollectorSensorConfig(StrictConfigModel):
    """Sensor parameters for the three Collector obstacle rays."""

    ray_length: float = Field(gt=0.0)
    ray_angles: list[float] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_ray_angles(self) -> Self:
        """Require ordered left, center, and right obstacle rays."""
        left_angle, center_angle, right_angle = self.ray_angles

        if center_angle != 0.0:
            raise ValueError("The center ray angle must be 0.0.")

        if not left_angle < center_angle < right_angle:
            raise ValueError(
                "ray_angles must contain ordered left, center, and right rays."
            )

        return self


class CollectorExplorationConfig(StrictConfigModel):
    """Configuration for the grid-based exploration bonus."""

    model_config = ConfigDict(extra="forbid")

    cell_size: int = Field(gt=0)


class CollectorTaskConfig(YamlConfigModel):
    """Complete YAML configuration for one Collector experiment."""

    env: CollectorEnvConfig
    reward: CollectorRewardConfig
    sensor: CollectorSensorConfig
    exploration: CollectorExplorationConfig
