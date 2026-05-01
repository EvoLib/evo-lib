# SPDX-License-Identifier: MIT
"""Checkpoint utilities for trained agents."""

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EnvSpec:
    """Specification required to recreate an environment."""

    name: str
    difficulty: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnvCheckpoint:
    """Serializable checkpoint of a trained individual."""

    indiv: Any
    env: EnvSpec
    seed: int | None = None


def save_checkpoint(path: str | Path, checkpoint: EnvCheckpoint) -> None:
    """Persist a checkpoint to disk."""

    with Path(path).open("wb") as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(path: str | Path) -> EnvCheckpoint:
    """Load a checkpoint from disk."""

    with Path(path).open("rb") as f:
        obj = pickle.load(f)

    if not isinstance(obj, EnvCheckpoint):
        raise TypeError(f"Expected EnvCheckpoint, got {type(obj).__name__}")

    return obj
