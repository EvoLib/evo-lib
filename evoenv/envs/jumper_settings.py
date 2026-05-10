# SPDX-License-Identifier: MIT
"""Difficulty presets for the Jumper environment."""

from dataclasses import dataclass
from enum import StrEnum


class JumperDifficulty(StrEnum):
    """Supported Jumper difficulty presets."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass(frozen=True)
class JumperSettings:
    """Configuration bundle for one Jumper difficulty preset."""

    difficulty: JumperDifficulty
    observation_size: int
    sensor_range: float
    variable_obstacle_height: bool
    variable_obstacle_width: bool
    allow_air_jump: bool
    min_obstacle_height: int
    max_obstacle_height: int
    min_obstacle_width: int
    max_obstacle_width: int


JUMPER_SETTINGS: dict[JumperDifficulty, JumperSettings] = {
    JumperDifficulty.EASY: JumperSettings(
        difficulty=JumperDifficulty.EASY,
        observation_size=2,
        sensor_range=220.0,
        variable_obstacle_height=False,
        variable_obstacle_width=False,
        allow_air_jump=False,
        min_obstacle_height=42,
        max_obstacle_height=42,
        min_obstacle_width=32,
        max_obstacle_width=32,
    ),
    JumperDifficulty.MEDIUM: JumperSettings(
        difficulty=JumperDifficulty.MEDIUM,
        observation_size=3,
        sensor_range=220.0,
        variable_obstacle_height=True,
        variable_obstacle_width=False,
        allow_air_jump=False,
        min_obstacle_height=24,
        max_obstacle_height=64,
        min_obstacle_width=32,
        max_obstacle_width=32,
    ),
    JumperDifficulty.HARD: JumperSettings(
        difficulty=JumperDifficulty.HARD,
        observation_size=5,
        sensor_range=220.0,
        variable_obstacle_height=True,
        variable_obstacle_width=True,
        allow_air_jump=True,
        min_obstacle_height=24,
        max_obstacle_height=72,
        min_obstacle_width=24,
        max_obstacle_width=56,
    ),
}


def get_jumper_settings(difficulty: str | JumperDifficulty) -> JumperSettings:
    """Return Jumper settings for a difficulty name."""

    return JUMPER_SETTINGS[JumperDifficulty(difficulty)]
