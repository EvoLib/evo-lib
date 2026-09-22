# SPDX-License-Identifier: MIT
"""Configuration models for the built-in Foraging simulation."""

import math
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from evolib.config.evonet_component_config import EvoNetComponentConfig
from evolib.config.vector_component_config import VectorComponentConfig


class WorldConfig(BaseModel):
    """World dimensions and population limits."""

    model_config = ConfigDict(extra="forbid")

    width: int = Field(default=900, gt=0)
    height: int = Field(default=600, gt=0)
    initial_population: int = Field(default=30, ge=1)
    max_population: int = Field(default=150, ge=1)

    @model_validator(mode="after")
    def validate_population_limit(self) -> Self:
        """Ensure the initial population fits into the configured world limit."""
        if self.initial_population > self.max_population:
            raise ValueError("initial_population must not exceed max_population.")
        return self


class FoodConfig(BaseModel):
    """Resource spawning and energy settings."""

    model_config = ConfigDict(extra="forbid")

    initial_count: int = Field(default=120, ge=0)
    max_count: int = Field(default=180, ge=0)
    spawn_rate: float = Field(default=0.7, ge=0.0)
    radius: float = Field(default=4.0, gt=0.0)
    energy: float = Field(default=24.0, gt=0.0)

    @model_validator(mode="after")
    def validate_food_limit(self) -> Self:
        """Ensure the initial food count fits below the configured limit."""
        if self.initial_count > self.max_count:
            raise ValueError("initial_count must not exceed max_count.")
        return self


class PoisonConfig(BaseModel):
    """Optional poison spawning and damage settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    initial_count: int = Field(default=20, ge=0)
    max_count: int = Field(default=60, ge=0)
    spawn_rate: float = Field(default=0.04, ge=0.0)
    radius: float = Field(default=4.0, gt=0.0)
    damage: float = Field(default=20.0, gt=0.0)

    @model_validator(mode="after")
    def validate_poison_limit(self) -> Self:
        """Ensure the initial poison count fits below the configured limit."""
        if self.initial_count > self.max_count:
            raise ValueError("initial_count must not exceed max_count.")
        return self


class ForagerConfig(BaseModel):
    """Forager body, movement, energy, and reproduction settings."""

    model_config = ConfigDict(extra="forbid")

    radius: float = Field(default=7.0, gt=0.0)
    max_speed: float = Field(default=2.5, ge=0.0)
    max_turn_rate: float = Field(default=0.22, ge=0.0)

    initial_energy: float = Field(default=55.0, gt=0.0)
    energy_capacity: float = Field(default=100, gt=0.0)
    basal_cost: float = Field(default=0.045, ge=0.0)
    movement_cost: float = Field(default=0.025, ge=0.0)
    turn_cost_factor: float = Field(default=1.0, ge=0.0)
    reverse_factor: float = Field(default=0.3, ge=0.0)
    feeding_cooldown_steps: int = Field(default=20, ge=0)

    reproduction_threshold: float = Field(default=90.0, gt=0.0)
    reproduction_cost: float = Field(default=34.0, gt=0.0)
    offspring_energy: float = Field(default=28.0, gt=0.0)
    min_reproduction_age_steps: int = Field(default=80, ge=0)
    max_age_steps: int = Field(default=5000, ge=0)

    @model_validator(mode="after")
    def validate_reproduction_energy(self) -> Self:
        """Validate the simple energy budget used for asexual reproduction."""
        if self.initial_energy > self.energy_capacity:
            raise ValueError("initial_energy must not exceed energy_capacity.")
        if self.offspring_energy > self.energy_capacity:
            raise ValueError("offspring_energy must not exceed energy_capacity.")
        if self.reproduction_threshold > self.energy_capacity:
            raise ValueError("reproduction_threshold must not exceed energy_capacity.")
        if self.reproduction_cost < self.offspring_energy:
            raise ValueError("reproduction_cost must be >= offspring_energy.")
        if self.reproduction_threshold <= self.reproduction_cost:
            raise ValueError("reproduction_threshold must exceed reproduction_cost.")
        return self


class MetricsConfig(BaseModel):
    """Periodic metric logging settings."""

    model_config = ConfigDict(extra="forbid")

    interval: int = Field(default=100, gt=0)
    file: str | None = None


class ForagingModulesConfig(BaseModel):
    """Evolvable modules carried by every Forager individual."""

    model_config = ConfigDict(extra="forbid")

    controller: EvoNetComponentConfig
    sensor_angles: VectorComponentConfig
    sensor_fovs: VectorComponentConfig
    sensor_ranges: VectorComponentConfig

    @model_validator(mode="after")
    def validate_sensor_modules(self) -> Self:
        """Validate Foraging-specific sensor and controller constraints."""
        angle_dim = self.sensor_angles.dim
        fov_dim = self.sensor_fovs.dim
        range_dim = self.sensor_ranges.dim

        if not (
            isinstance(angle_dim, int)
            and isinstance(fov_dim, int)
            and isinstance(range_dim, int)
        ):
            raise ValueError(
                "Foraging sensor vectors must use flat integer dimensions."
            )

        sensor_count = angle_dim

        if not (fov_dim == sensor_count and range_dim == sensor_count):
            raise ValueError(
                "Sensor angle, FOV, and range vectors must have equal size."
            )

        fov_low, fov_high = self.sensor_fovs.bounds
        if fov_low <= 0.0 or fov_high > math.tau:
            raise ValueError("Sensor FOV bounds must lie within (0, 2*pi].")

        range_low, _ = self.sensor_ranges.bounds
        if range_low <= 0.0:
            raise ValueError("Sensor range bounds must be greater than zero.")

        if self.controller.dim[-1] != 2:
            raise ValueError("Foraging requires two EvoNet output neurons.")

        return self

    @property
    def sensor_count(self) -> int:
        """Return the number of configured local sensors."""
        dim = self.sensor_angles.dim
        if not isinstance(dim, int):
            raise ValueError("Foraging sensor vectors require integer dimensions.")
        return dim


class ForagingConfig(BaseModel):
    """Complete configuration for the built-in Foraging simulation."""

    model_config = ConfigDict(extra="forbid")

    seed: int | None = 1
    max_steps: int = Field(default=100_000, gt=0)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    world: WorldConfig = Field(default_factory=WorldConfig)
    food: FoodConfig = Field(default_factory=FoodConfig)
    poison: PoisonConfig = Field(default_factory=PoisonConfig)
    forager: ForagerConfig = Field(default_factory=ForagerConfig)
    modules: ForagingModulesConfig

    @model_validator(mode="after")
    def validate_controller_inputs(self) -> Self:
        """Validate controller inputs for the enabled sensory channels."""
        channels_per_sensor = 2 if self.poison.enabled else 1
        expected_inputs = self.modules.sensor_count * channels_per_sensor + 2
        if self.modules.controller.dim[0] != expected_inputs:
            raise ValueError(
                f"Foraging requires {expected_inputs} EvoNet inputs for the "
                "configured sensory and state inputs."
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        """Load and validate a Foraging configuration from YAML."""
        config_path = Path(path)
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Foraging configuration must contain a YAML mapping.")
        return cls.model_validate(data)
