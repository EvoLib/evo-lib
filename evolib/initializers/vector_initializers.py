# SPDX-License-Identifier: MIT
"""Vector initializer adapters delegating to Vector.from_config()."""

from typing import Literal

from evolib.config.schema import FullConfig
from evolib.config.vector_component_config import VectorComponentConfig
from evolib.representation.vector import Vector

VectorInitializer = Literal["normal", "uniform", "zero", "fixed", "adaptive"]


def _initialize_vector(
    config: FullConfig,
    module: str,
    initializer: VectorInitializer,
) -> Vector:
    """Initialize one Vector module through Vector.from_config()."""
    cfg = config.modules[module]
    if not isinstance(cfg, VectorComponentConfig):
        raise TypeError(f"Module '{module}' is not a Vector module")

    if cfg.initializer != initializer:
        raise ValueError(
            f"Module '{module}' uses initializer {cfg.initializer!r}; "
            f"expected {initializer!r}."
        )

    return Vector.from_config(cfg)


def initializer_normal_vector(config: FullConfig, module: str) -> Vector:
    """Initialize a Vector using a normal distribution."""
    return _initialize_vector(config, module, "normal")


def initializer_random_vector(config: FullConfig, module: str) -> Vector:
    """Initialize a Vector using a uniform distribution."""
    return _initialize_vector(config, module, "uniform")


def initializer_zero_vector(config: FullConfig, module: str) -> Vector:
    """Initialize a Vector with all zeros."""
    return _initialize_vector(config, module, "zero")


def initializer_fixed_vector(config: FullConfig, module: str) -> Vector:
    """Initialize a Vector with fixed values from the config."""
    return _initialize_vector(config, module, "fixed")


def initializer_adaptive_vector(config: FullConfig, module: str) -> Vector:
    """Initialize a Vector with adaptive per-parameter mutation state."""
    return _initialize_vector(config, module, "adaptive")
