# SPDX-License-Identifier: MIT
"""Configuration models for the built-in Predator-Prey simulation."""

import math
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from evolib.config.evonet_component_config import EvoNetComponentConfig


class WorldConfig(BaseModel):
    """World dimensions."""

    model_config = ConfigDict(extra="forbid")

    width: int = Field(default=900, gt=0)
    height: int = Field(default=600, gt=0)


class PreyConfig(BaseModel):
    """Prey movement, grazing, reproduction, and population settings."""

    model_config = ConfigDict(extra="forbid")

    initial_population: int = Field(default=100, ge=1)
    max_population: int = Field(default=160, ge=1)

    radius: float = Field(default=7.0, gt=0.0)
    max_speed: float = Field(default=2.5, ge=0.0)
    max_turn_rate: float = Field(default=0.22, ge=0.0)
    reverse_factor: float = Field(default=0.3, ge=0.0)

    energy_capacity: float = Field(default=100.0, gt=0.0)
    movement_cost: float = Field(default=0.005, ge=0.0)
    grazing_rate: float = Field(default=0.12, ge=0.0)
    grazing_max_speed: float = Field(default=0.1, ge=0.0)

    reproduction_interval_steps: int = Field(default=700, gt=0)
    reproduction_interval_jitter_steps: int = Field(default=100, ge=0)
    offspring_dispersion: float = Field(default=100.0, gt=0.0)

    energy_value: float = Field(default=30.0, gt=0.0)

    @model_validator(mode="after")
    def validate_prey(self) -> Self:
        """Validate Prey population, grazing, and reproduction settings."""
        if self.initial_population > self.max_population:
            raise ValueError("initial_population must not exceed max_population.")
        if self.grazing_max_speed > self.max_speed:
            raise ValueError("grazing_max_speed must not exceed max_speed.")
        if self.reproduction_interval_jitter_steps >= self.reproduction_interval_steps:
            raise ValueError(
                "reproduction_interval_jitter_steps must be smaller than "
                "reproduction_interval_steps."
            )
        if self.offspring_dispersion < self.radius * 3.0:
            raise ValueError("offspring_dispersion must be at least three radii.")
        return self


class PredatorConfig(BaseModel):
    """Predator movement, energy, feeding, and reproduction settings."""

    model_config = ConfigDict(extra="forbid")

    initial_population: int = Field(default=20, ge=1)
    max_population: int = Field(default=32, ge=1)

    radius: float = Field(default=7.0, gt=0.0)
    max_speed: float = Field(default=2.5, ge=0.0)
    max_turn_rate: float = Field(default=0.11, ge=0.0)
    reverse_factor: float = Field(default=0.3, ge=0.0)

    initial_energy: float = Field(default=50.0, gt=0.0)
    energy_capacity: float = Field(default=100.0, gt=0.0)
    basal_cost: float = Field(default=0.025, ge=0.0)
    movement_cost: float = Field(default=0.05, ge=0.0)
    feeding_cooldown_steps: int = Field(default=100, ge=0)
    reproduction_cooldown_steps: int = Field(default=300, ge=0)
    offspring_dispersion: float = Field(default=100.0, gt=0.0)

    @model_validator(mode="after")
    def validate_predator(self) -> Self:
        """Validate Predator population and energy settings."""
        if self.initial_population > self.max_population:
            raise ValueError("initial_population must not exceed max_population.")
        if self.initial_energy > self.energy_capacity:
            raise ValueError("initial_energy must not exceed energy_capacity.")
        if self.offspring_dispersion < self.radius * 3.0:
            raise ValueError("offspring_dispersion must be at least three radii.")
        return self


class MetricsConfig(BaseModel):
    """Periodic metric logging settings."""

    model_config = ConfigDict(extra="forbid")

    interval: int = Field(default=100, gt=0)
    file: str | None = None


class AgentModulesConfig(BaseModel):
    """Controller and fixed ray-sensor settings for one population."""

    model_config = ConfigDict(extra="forbid")

    controller: EvoNetComponentConfig
    ray_count: int = Field(default=19, ge=1)
    sensor_fov: float = Field(gt=0.0, le=math.tau)
    sensor_range: float = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_controller(self) -> Self:
        """Validate the controller outputs used for movement."""
        if self.controller.dim[-1] != 2:
            raise ValueError("Predator-Prey requires two EvoNet output neurons.")
        return self


class ModulesConfig(BaseModel):
    """Separate controller and sensor settings for Prey and Predators."""

    model_config = ConfigDict(extra="forbid")

    prey: AgentModulesConfig
    predator: AgentModulesConfig


class PredatorPreyConfig(BaseModel):
    """Complete configuration for the built-in Predator-Prey simulation."""

    model_config = ConfigDict(extra="forbid")

    seed: int | None = 1
    max_steps: int = Field(default=100_000, gt=0)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    world: WorldConfig = Field(default_factory=WorldConfig)
    prey: PreyConfig = Field(default_factory=PreyConfig)
    predator: PredatorConfig = Field(default_factory=PredatorConfig)
    modules: ModulesConfig

    @model_validator(mode="after")
    def validate_controller_inputs(self) -> Self:
        """Validate ray channels and normalized internal-state inputs."""
        prey_inputs = self.modules.prey.ray_count + 1
        predator_inputs = self.modules.predator.ray_count + 2

        if self.modules.prey.controller.dim[0] != prey_inputs:
            raise ValueError(
                f"Prey requires {prey_inputs} EvoNet inputs for Predator rays "
                "and energy."
            )
        if self.modules.predator.controller.dim[0] != predator_inputs:
            raise ValueError(
                f"Predator requires {predator_inputs} EvoNet inputs for Prey rays, "
                "energy, and feeding cooldown."
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        """Load and validate a Predator-Prey configuration from YAML."""
        config_path = Path(path)
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Predator-Prey configuration must contain a YAML mapping.")
        return cls.model_validate(data)
