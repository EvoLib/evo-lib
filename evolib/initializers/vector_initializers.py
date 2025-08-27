# SPDX-License-Identifier: MIT
"""
Initializers for Vector representations.

These functions use a full configuration (FullConfig) and a module name to initialize
Vector instances directly.

Each initializer is compatible with EvoLib’s mutation, crossover, and adaptation system.
"""

import numpy as np

from evolib.config.schema import FullConfig
from evolib.representation.vector import Vector


def initializer_normal_vector(config: FullConfig, module: str) -> Vector:
    """
    Initializes a Vector using a normal distribution. init_bounds act as a hard clamp
    for initializer outputs; if omitted, bounds are used as fallback.

    Args:
        config (FullConfig): Full config object containing all module definitions
        module (str): Name of the module (e.g. "weights")

    Returns:
        Vector: Initialized vector
    """
    para = Vector()
    cfg = config.modules[module]
    para.apply_config(cfg)

    vec = np.random.normal(
        loc=cfg.mean or 0.0,
        scale=cfg.std or 1.0,
        size=para.dim,
    )

    # clamp to init_bounds (fallback to bounds)
    bounds = para.init_bounds or para.bounds
    if bounds is not None:
        lo, hi = bounds
        vec = np.clip(vec, lo, hi)

    para.vector = vec
    return para


def initializer_random_vector(config: FullConfig, module: str) -> Vector:
    """
    Initializes a Vector with uniform random values from init_bounds.

    Args:
        config (FullConfig): Full config object
        module (str): Module name

    Returns:
        Vector: Initialized vector
    """
    para = Vector()
    cfg = config.modules[module]
    para.apply_config(cfg)

    if para.init_bounds is None:
        raise ValueError(f"init_bounds must be set for module '{module}'")

    size = int(para.dim)
    para.vector = np.random.uniform(*para.init_bounds, size=size)
    return para


def initializer_zero_vector(config: FullConfig, module: str) -> Vector:
    """
    Initializes a Vector with all zeros.

    Args:
        config (FullConfig): Full config object
        module (str): Module name

    Returns:
        Vector: Initialized vector
    """
    para = Vector()
    cfg = config.modules[module]
    para.apply_config(cfg)

    para.vector = np.zeros(para.dim)
    return para


def initializer_fixed_vector(config: FullConfig, module: str) -> Vector:
    """
    Initializes a Vector with fixed values from the config.

    Args:
        config (FullConfig): Full config object
        module (str): Module name

    Returns:
        Vector: Initialized vector
    """
    para = Vector()
    cfg = config.modules[module]
    para.apply_config(cfg)

    if cfg.values is None:
        raise ValueError(
            f"values must be defined for initializer_fixed_vector "
            f"in module '{module}'"
        )

    para.vector = np.array(cfg.values)
    return para


def initializer_adaptive_vector(config: FullConfig, module: str) -> Vector:
    """
    Initializes a Vector with random values and per-parameter mutation strengths.

    Args:
        config (FullConfig): Full config object
        module (str): Module name

    Returns:
        Vector: Initialized vector
    """
    para = Vector()
    cfg = config.modules[module]
    para.apply_config(cfg)

    if para.init_bounds is None:
        raise ValueError(f"init_bounds must be defined in module '{module}'")
    if (
        para.evo_params.min_mutation_strength is None
        or para.evo_params.max_mutation_strength is None
    ):
        raise ValueError(
            f"min/max mutation strength must be defined in " f"module '{module}'"
        )

    size = int(para.dim)
    para.vector = np.random.uniform(*para.init_bounds, size=size)

    if para.randomize_mutation_strengths:
        para.evo_params.mutation_strengths = np.random.uniform(
            para.evo_params.min_mutation_strength,
            para.evo_params.max_mutation_strength,
            size=para.dim,
        )
    else:
        if para.evo_params.mutation_strength is None:
            raise ValueError(
                f"mutation_strength must be defined for non-random initialization "
                f"in module '{module}'"
            )
        para.evo_params.mutation_strengths = np.full(
            para.dim, para.evo_params.mutation_strength
        )

    return para
