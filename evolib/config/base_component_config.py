# SPDX-License-Identifier: MIT
"""
Shared configuration blocks for mutation and crossover used across multiple
ComponentConfig classes (e.g. VectorComponentConfig, EvoNetComponentConfig).

The classes here are intentionally small and reusable: they describe *what*
should be configured, not *how* it is executed. Any runtime behavior belongs
into the respective Para* representations and operator modules.
"""

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

    Supported strategies (see MutationStrategy enum):
        - CONSTANT
        - EXPONENTIAL_DECAY
        - ADAPTIVE_GLOBAL
        - ADAPTIVE_INDIVIDUAL
        - ADAPTIVE_PER_PARAMETER

    Which fields are relevant depends on the selected strategy:

    CONSTANT
        - strength (required)
        - probability (optional; default behavior handled downstream)

    EXPONENTIAL_DECAY
        - init_strength (required)
        - init_probability (optional)

    ADAPTIVE_GLOBAL
        - strength (required as starting point; mapped to runtime state)
        - probability (required as starting point)
        - increase_factor / decrease_factor (optional)
        - min_diversity_threshold / max_diversity_threshold (optional)
        - min_strength / max_strength (optional clamp)
        - min_probability / max_probability (optional clamp)

    ADAPTIVE_INDIVIDUAL / ADAPTIVE_PER_PARAMETER
        - min_strength, max_strength (required range for sigma updates)
        - probability (optional)
        - increase_factor / decrease_factor, diversity thresholds (optional)
        - min_probability / max_probability (optional clamp)

    This class is purely declarative. Strategy-specific calculations and
    state updates occur in the corresponding Para* implementations or update
    helpers.
    """

    # Mutation strategy to use
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


class EvoNetMutationConfig(MutationConfig):
    """
    EvoNet-specific mutation configuration with optional per-scope overrides.

    If an override is omitted, the base fields of this config apply.
    """

    biases: Optional[MutationConfig] = None


class CrossoverConfig(BaseModel):
    """
    Configuration block for crossover strategies and operators.

    Strategy (high-level policy) and Operator (low-level mechanism) are modeled
    separately. Depending on the operator, additional parameters may apply.

    Operators (see CrossoverOperator):
        - BLX (uses alpha)
        - SBX (uses eta)
        - INTERMEDIATE (uses blend_range)
        - ...
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
