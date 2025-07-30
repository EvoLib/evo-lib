# SPDX-License-Identifier: MIT
"""Shared mutation and crossover configuration schemas used by multiple ComponentConfig
classes (e.g., VectorComponentConfig, EvoNetComponentConfig)."""

from typing import Optional

from pydantic import BaseModel

from evolib.interfaces.enums import (
    CrossoverOperator,
    CrossoverStrategy,
    MutationStrategy,
)


class MutationConfig(BaseModel):
    """
    Configuration block for mutation strategies.

    Supports constant, exponential, adaptive global/individual/per-parameter mutation.
    Only required fields need to be set based on strategy.
    """

    strategy: MutationStrategy

    # For constant mutation
    strength: Optional[float] = None
    probability: Optional[float] = None

    # For exponential and adaptive strategies
    init_strength: Optional[float] = None
    init_probability: Optional[float] = None

    min_strength: Optional[float] = None
    max_strength: Optional[float] = None

    min_probability: Optional[float] = None
    max_probability: Optional[float] = None

    # Diversity adaptation
    increase_factor: Optional[float] = None
    decrease_factor: Optional[float] = None
    min_diversity_threshold: Optional[float] = None
    max_diversity_threshold: Optional[float] = None


class CrossoverConfig(BaseModel):
    """
    Configuration block for crossover strategies.

    Supports constant, exponential, adaptive crossover, and several operators.
    """

    strategy: CrossoverStrategy
    operator: Optional[CrossoverOperator] = None

    # Probability settings
    probability: Optional[float] = None
    init_probability: Optional[float] = None
    min_probability: Optional[float] = None
    max_probability: Optional[float] = None

    # Diversity adaptation
    increase_factor: Optional[float] = None
    decrease_factor: Optional[float] = None

    # Operator-specific parameters
    alpha: Optional[float] = None  # for BLX
    eta: Optional[float] = None  # for SBX
    blend_range: Optional[float] = None  # for intermediate crossover
