# SPDX-License-Identifier: MIT
"""Helpers for difficulty-based example file names."""

from enum import StrEnum
from pathlib import Path


class Difficulty(StrEnum):
    """Shared difficulty presets for EvoEnv examples."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


def difficulty_config_path(
    difficulty: str | Difficulty,
    *,
    directory: str | Path = ".",
) -> Path:
    """Return the EvoLib config path for a difficulty preset."""
    name = Difficulty(difficulty).value
    return Path(directory) / f"config_{name}.yaml"


def difficulty_checkpoint_path(
    env_name: str,
    difficulty: str | Difficulty,
    *,
    directory: str | Path = ".",
) -> Path:
    """Return the checkpoint path for an environment difficulty preset."""
    name = Difficulty(difficulty).value
    return Path(directory) / f"{env_name}_{name}.pkl"


def difficulty_task_path(
    difficulty: str | Difficulty,
    *,
    directory: str | Path = ".",
) -> Path:
    """Return the task config path for a difficulty preset."""
    name = Difficulty(difficulty).value
    return Path(directory) / f"task_{name}.yaml"
