# SPDX-License-Identifier: MIT
"""Difficulty presets for the LineFollower environment."""

from dataclasses import dataclass
from enum import StrEnum


class LineFollowerDifficulty(StrEnum):
    """Supported LineFollower difficulty presets."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass(frozen=True)
class LineFollowerSettings:
    """
    Configuration bundle for one LineFollower difficulty preset.

    Note:
        Observation size remains constant (2 sensors) across all difficulties.
        Difficulty is controlled via environment dynamics and tolerance.
    """

    difficulty: LineFollowerDifficulty

    # Environment shape
    line_complexity: float
    line_width: int

    # Robot behavior
    base_speed: float
    turn_strength: float

    # Termination tolerance
    max_missed_line_steps: int


LINE_FOLLOWER_SETTINGS: dict[LineFollowerDifficulty, LineFollowerSettings] = {
    LineFollowerDifficulty.EASY: LineFollowerSettings(
        difficulty=LineFollowerDifficulty.EASY,
        line_complexity=1.2,  # nearly straight
        line_width=10,  # wide line → easy detection
        base_speed=3.5,  # slower → easier control
        turn_strength=0.10,
        max_missed_line_steps=60,  # forgiving
    ),
    LineFollowerDifficulty.MEDIUM: LineFollowerSettings(
        difficulty=LineFollowerDifficulty.MEDIUM,
        line_complexity=2.5,
        line_width=6,
        base_speed=4.5,
        turn_strength=0.12,
        max_missed_line_steps=35,
    ),
    LineFollowerDifficulty.HARD: LineFollowerSettings(
        difficulty=LineFollowerDifficulty.HARD,
        line_complexity=3.8,  # strong curves
        line_width=4,  # narrow -> harder sensing
        base_speed=5.5,  # faster -> harder control
        turn_strength=0.14,
        max_missed_line_steps=20,  # strict
    ),
}


def get_line_follower_settings(
    difficulty: str | LineFollowerDifficulty,
) -> LineFollowerSettings:
    """Return LineFollower settings for a difficulty name."""

    return LINE_FOLLOWER_SETTINGS[LineFollowerDifficulty(difficulty)]
