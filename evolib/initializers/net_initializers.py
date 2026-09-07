# SPDX-License-Identifier: MIT
"""Adapter for Vector-based NetVector initialization."""

from evolib.config.schema import FullConfig
from evolib.config.vector_component_config import VectorComponentConfig
from evolib.representation.vector import Vector


def initializer_normal_net(config: FullConfig, module: str) -> Vector:
    """Initialize a Vector using its ``structure: net`` configuration."""
    cfg = config.modules[module]
    if not isinstance(cfg, VectorComponentConfig):
        raise TypeError(f"Module '{module}' is not a Vector module")

    if cfg.initializer != "normal":
        raise ValueError(
            f"Module '{module}' uses initializer {cfg.initializer!r}; "
            "expected 'normal'."
        )
    if cfg.structure != "net":
        raise ValueError(f"Module '{module}' must use structure='net'.")

    return Vector.from_config(cfg)
