# SPDX-License-Identifier: MIT
"""Helpers for difficulty-based example file names."""

from pathlib import Path


def difficulty_config_path(difficulty: str, *, directory: str | Path = ".") -> Path:
    """Return the config path for a difficulty preset."""

    return Path(directory) / f"config_{difficulty}.yaml"


def difficulty_checkpoint_path(
    env_name: str,
    difficulty: str,
    *,
    directory: str | Path = ".",
) -> Path:
    """Return the checkpoint path for an environment difficulty preset."""

    return Path(directory) / f"{env_name}_{difficulty}.pkl"
