# SPDX-License-Identifier: MIT
"""
Low-level serialization helpers (pickle + optional gzip).

Kept separate from checkpoint semantics for clarity and testability.
"""

from __future__ import annotations

import gzip
import pickle
from pathlib import Path
from typing import Any


def write_pickle(obj: Any, path: str | Path, *, compressed: bool | None = None) -> str:
    """
    Write an object via pickle to disk.

    Args:
        obj: Any Python object to serialize.
        path: Target file path ('.pkl' or '.pkl.gz').
        compressed: If True, force gzip; if False, force plain.
                    If None, infer from file suffix.

    Returns:
        str: The absolute path written to.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    use_gzip = compressed if compressed is not None else p.suffix.endswith("gz")

    if use_gzip:
        with gzip.open(p, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with p.open("wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return str(p.resolve())


def read_pickle(path: str | Path) -> Any:
    """
    Read an object via pickle from disk.

    Gzip is auto-detected by suffix '.gz'.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    if p.suffix.endswith("gz"):
        with gzip.open(p, "rb") as f:
            return pickle.load(f)
    else:
        with p.open("rb") as f:
            return pickle.load(f)


def save_population_pickle(pop: Any, path: str | Path) -> str:
    """Serialize a Population object."""
    return write_pickle(pop, path)


def load_population_pickle(path: str | Path) -> Any:
    """Deserialize a Population object."""
    return read_pickle(path)


def save_indiv(indiv: Any, path: str | Path) -> str:
    """Serialize a single individual."""
    return write_pickle(indiv, path)


def load_indiv(path: str | Path) -> Any:
    """Deserialize a single individual."""
