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

from typing import TYPE_CHECKING, List

import numpy as np

if TYPE_CHECKING:
    from evolib.core.population import Pop

from evolib.core.population import Indiv
from evolib.interfaces.types import MutationParams


def mutate_offspring(
    pop: "Pop",
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
