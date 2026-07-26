# SPDX-License-Identifier: MIT
"""Pydantic configuration models for the LineFollower example."""

from __future__ import annotations

from evoenv.core.config import StrictConfigModel, YamlConfigModel
from pydantic import Field


class LineFollowerEnvConfig(StrictConfigModel):
    """Simulation parameters for the LineFollower environment."""

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    max_steps: int = Field(gt=0)

    line_complexity: float = Field(gt=0.0)
    line_width: int = Field(gt=0)

    base_speed: float = Field(gt=0.0)
    turn_strength: float = Field(gt=0.0)

    max_missed_line_steps: int = Field(gt=0)


class LineFollowerRewardConfig(StrictConfigModel):
    """Reward parameters for the LineFollower task."""

    progress_reward_scale: float = Field(ge=0.0)
    missed_line_penalty: float = Field(ge=0.0)


class LineFollowerTaskConfig(YamlConfigModel):
    """Complete YAML configuration for one LineFollower difficulty preset."""

    env: LineFollowerEnvConfig
    reward: LineFollowerRewardConfig
