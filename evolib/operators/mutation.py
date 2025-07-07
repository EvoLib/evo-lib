# SPDX-License-Identifier: MIT
"""
Provides mutation utilities for evolutionary strategies.

This module defines functions to apply mutations to individuals or entire offspring
populations, based on configurable mutation strategies (e.g., exponential, adaptive).
It delegates actual parameter mutation to user-defined mutation functions.

Functions:
- mutate_indiv: Mutates a single individual based on the population's strategy.
- mutate_offspring: Mutates all individuals in an offspring list.

Expected mutation functions must operate on the parameter level and implement
mutation probability checks internally.
"""

from typing import List, Optional

import numpy as np

from evolib.core.population import Indiv, Pop
from evolib.interfaces.enums import MutationStrategy
from evolib.interfaces.types import MutationParams


def mutate_offspring(
    pop: Pop,
    offspring: List[Indiv],
) -> None:
    """
    Applies mutation to all individuals in the offspring list.

    Args:
        pop (Pop): The population object containing mutation configuration.
        offspring (List[Indiv]): List of individuals to mutate.
    """

    for indiv in offspring:
        indiv.mutate()


def get_mutation_parameters(indiv: Indiv) -> tuple[float, float]:
    """
    Retrieve the effective mutation parameters (rate, strength) for a given individual.

    Args:
        indiv (Indiv): The individual whose mutation parameters may be used.

    Returns:
        tuple[float, float]: (mutation_probability, mutation_strength)

    Raises:
        ValueError: If required individual parameters are missing or strategy is
        unsupported.
    """
    strategy = indiv.para.mutation_strategy

    if strategy in {
        MutationStrategy.CONSTANT,
        MutationStrategy.EXPONENTIAL_DECAY,
        MutationStrategy.ADAPTIVE_GLOBAL,
    }:
        return indiv.para.mutation_probability, indiv.para.mutation_strength

    if strategy == MutationStrategy.ADAPTIVE_INDIVIDUAL:
        if (
            indiv.para.mutation_probability is None
            or indiv.para.mutation_strength is None
        ):
            raise ValueError(
                "Individual mutation parameters must be initialized before use "
                "when using ADAPTIVE_INDIVIDUAL strategy."
            )
        return indiv.para.mutation_probability, indiv.para.mutation_strength

    if strategy == MutationStrategy.ADAPTIVE_PER_PARAMETER:
        # Use average of per-parameter strengths, fallback 0.0
        strengths = indiv.para.mutation_strengths
        avg_strength = sum(strengths) / len(strengths) if strengths else 0.0
        return indiv.para.mutation_probability or 0.0, avg_strength

    raise ValueError(f"Unsupported mutation strategy: {strategy}")


def mutate_gauss(
    x: np.ndarray,
    mutation_strength: Optional[float] = 0.05,
    bounds: tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    """
    Mutates a scalar or vector value by adding Gaussian noise.

    Args:
        x (np.ndarray): Input value or parameter vector.
        mutation_strength (float | None): Strength of the Gaussian noise.
            May be None if unset (e.g., before adaptive mutation initialization),
            in which case a ValueError is raised. This allows flexible typing in
            user code that passes uninitialized individuals.
        bounds (tuple): Lower and upper clipping bounds.

    Raises:
        ValueError: If mutation_strength is None or not positive.
    """
    # Validate inputs
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
        # raise ValueError("Input x must be a NumPy array")
    if mutation_strength is None or mutation_strength <= 0:
        raise ValueError("mutation_strength must be a positive float")
    if not (isinstance(bounds, tuple) and len(bounds) == 2 and bounds[0] <= bounds[1]):
        raise ValueError("bounds must be a tuple (min, max) with min <= max")

    # Generate Gaussian noise with the same shape as x
    noise = np.random.normal(0, mutation_strength, size=x.shape)

    # Add noise and clip to bounds
    mutated = x + noise
    mutated = np.clip(mutated, bounds[0], bounds[1])

    return mutated


def adapted_mutation_strength(params: MutationParams) -> float:
    """
    Applies log-normal scaling and clipping to an individual's mutation_strength.

    Args:
        indiv (Indiv): The individual to update.
        params (MutationParams): Contains tau, min/max strength, etc.

    Returns:
        float: The updated mutation strength.
    """
    if params.tau is None:
        raise ValueError("tau must not be None for adaptive mutation strength")

    adapted = params.strength * np.exp(params.tau * np.random.normal())
    return float(np.clip(adapted, params.min_strength, params.max_strength))


def adapted_mutation_probability(params: MutationParams) -> float:
    """
    Applies log-normal scaling and clipping to an individual's mutation_probability.

    Args:
        indiv (Indiv): The individual to update.
        params (MutationParams): Contains tau, min/max strength, etc.

    Returns:
        float: The updated mutation probability.
    """

    if params.tau is None:
        raise ValueError("tau must not be None for adaptive mutation strength")

    adapted = params.probability * np.exp(params.tau * np.random.normal())
    return float(np.clip(adapted, params.min_probability, params.max_probability))
