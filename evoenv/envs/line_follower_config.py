# SPDX-License-Identifier: MIT
"""Pydantic configuration models for the LineFollower example."""

from __future__ import annotations

from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field


class LineFollowerEnvConfig(BaseModel):
    """Simulation parameters for the LineFollower environment."""

    model_config = ConfigDict(extra="forbid")

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    max_steps: int = Field(gt=0)

    line_complexity: float = Field(gt=0.0)
    line_width: int = Field(gt=0)

    base_speed: float = Field(gt=0.0)
    turn_strength: float = Field(gt=0.0)

    max_missed_line_steps: int = Field(gt=0)


class LineFollowerRewardConfig(BaseModel):
    """Reward parameters for the LineFollower task."""

    model_config = ConfigDict(extra="forbid")

    progress_reward_scale: float = Field(ge=0.0)
    missed_line_penalty: float = Field(ge=0.0)


class LineFollowerTaskConfig(BaseModel):
    """Complete YAML configuration for one LineFollower difficulty preset."""

    model_config = ConfigDict(extra="forbid")

    env: LineFollowerEnvConfig
    reward: LineFollowerRewardConfig

    def to_yaml_dict(self) -> dict[str, object]:
        """Return a YAML-serializable representation of the configuration."""
        return self.model_dump(mode="json")

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        """Load and validate a LineFollower task configuration from YAML."""
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as file:
            raw_config = yaml.safe_load(file) or {}

        return cls.model_validate(raw_config)
