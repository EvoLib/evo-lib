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

from typing import TYPE_CHECKING, List, cast

import numpy as np

if TYPE_CHECKING:
    from evolib.core.population import Pop

from evolib.core.individual import Indiv
from evolib.representation.evo_params import EvoControlParams


def adapted_tau(vector_length: int) -> float:
    """The learning rate tau based on the vector length."""
    return 1.0 / np.sqrt(vector_length) if vector_length > 0 else 0.0


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


def _adapted_mutation_strength_core(
    current: float | np.ndarray, tau: float, bounds: tuple[float, float]
) -> float | np.ndarray:
    """
    Applies log-normal self-adaptation to a scalar or vector of mutation strengths.

    Args:
        current: The current mutation strength(s), either scalar or array.
        tau: Learning rate for the mutation strength update.
        bounds: Tuple (min, max) for clipping the result.

    Returns:
        Updated and clipped mutation strength(s), matching the shape of `current`.
    """

    if isinstance(current, (float, int)):
        eps = np.random.normal()
    else:
        eps = np.random.normal(size=current.shape)

    updated = current * np.exp(tau * eps)

    return np.clip(updated, *bounds)


def adapt_mutation_strength(
    params: EvoControlParams, bounds: tuple[float, float]
) -> float:
    """Returns the updated scalar mutation strength using log-normal self-adaptation."""

    if params.mutation_strength is None:
        raise ValueError("mutation_strength must not be None")

    if params.tau is None:
        raise ValueError("tau must not be None for adaptive mutation strength")

    return float(
        _adapted_mutation_strength_core(
            current=params.mutation_strength, tau=params.tau, bounds=bounds
        )
    )


def adapt_mutation_strengths(
    params: EvoControlParams, bounds: tuple[float, float]
) -> np.ndarray:
    """Applies log-normal self-adaptation to per-parameter mutation strengths."""

    if params.mutation_strengths is None:
        raise ValueError("mutation_strength must not be None")

    if params.tau is None:
        raise ValueError("tau must not be None for adaptive mutation strength")

    return cast(
        np.ndarray,
        _adapted_mutation_strength_core(
            current=params.mutation_strengths, tau=params.tau, bounds=bounds
        ),
    )


def adapt_mutation_probability(params: EvoControlParams) -> float:
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

    adapted = params.mutation_probability * np.exp(params.tau * np.random.normal())
    return float(
        np.clip(
            adapted, params.min_mutation_probability, params.max_mutation_probability
        )
    )


def adapt_crossover_probability(params: EvoControlParams) -> float:
    """
    Applies log-normal scaling and clipping to an individual's crossover_probability.

    Args:
        indiv (Indiv): The individual to update.
        params (MutationParams): Contains tau, min/max strength, etc.

    Returns:
        float: The updated crossover probability.
    """

    if params.tau is None:
        raise ValueError("tau must not be None for adaptive mutation strength")

    adapted = params.crossover_probability * np.exp(params.tau * np.random.normal())
    return float(
        np.clip(
            adapted, params.min_crossover_probability, params.max_crossover_probability
        )
    )


def adapt_value_by_diversity(
    value: float,
    diversity_ema: float,
    min_threshold: float,
    max_threshold: float,
    increase_factor: float,
    decrease_factor: float,
    min_value: float,
    max_value: float,
) -> float:
    """
    Generic diversity-feedback-based parameter adaptation.

    Args:
        value: The current value (e.g. mutation strength).
        diversity_ema: Diversity measure of the population.
        min_threshold: Lower diversity bound triggering increase.
        max_threshold: Upper diversity bound triggering decrease.
        increase_factor: Multiplicative increase factor.
        decrease_factor: Multiplicative decrease factor.
        min_value: Minimum allowed value.
        max_value: Maximum allowed value.

    Returns:
        The adjusted and clipped value.
    """
    if diversity_ema < min_threshold:
        return min(max_value, value * increase_factor)
    elif diversity_ema > max_threshold:
        return max(min_value, value * decrease_factor)
    else:
        return value


def adapt_mutation_strength_by_diversity(
    value: float, diversity_ema: float, params: EvoControlParams
) -> float:
    """
    Adjusts mutation strength based on diversity using threshold-based scaling.

    Increases or decreases the value depending on whether diversity falls below or above
    configured thresholds in the control parameters.
    """

    if params.min_diversity_threshold is None:
        raise ValueError("min_diversity_threshold must not be None")
    if params.max_diversity_threshold is None:
        raise ValueError("max_diversity_threshold must not be None")
    if params.mutation_inc_factor is None:
        raise ValueError("mutation_inc_factor must not be None")
    if params.mutation_dec_factor is None:
        raise ValueError("mutation_dec_factor must not be None")
    if params.min_mutation_strength is None:
        raise ValueError("min_mutation_strength must not be None")
    if params.max_mutation_strength is None:
        raise ValueError("max_mutation_strength must not be None")

    return adapt_value_by_diversity(
        value=value,
        diversity_ema=diversity_ema,
        min_threshold=params.min_diversity_threshold,
        max_threshold=params.max_diversity_threshold,
        increase_factor=params.mutation_inc_factor,
        decrease_factor=params.mutation_dec_factor,
        min_value=params.min_mutation_strength,
        max_value=params.max_mutation_strength,
    )


def adapt_mutation_probability_by_diversity(
    value: float, diversity_ema: float, params: EvoControlParams
) -> float:
    """
    Adjusts mutation probability based on diversity using threshold-based scaling.

    Increases or decreases the value depending on whether diversity falls below or above
    configured thresholds in the control parameters.
    """

    if params.min_diversity_threshold is None:
        raise ValueError("min_diversity_threshold must not be None")
    if params.max_diversity_threshold is None:
        raise ValueError("max_diversity_threshold must not be None")
    if params.mutation_inc_factor is None:
        raise ValueError("mutation_inc_factor must not be None")
    if params.mutation_dec_factor is None:
        raise ValueError("mutation_dec_factor must not be None")
    if params.min_mutation_probability is None:
        raise ValueError("min_mutation_probability must not be None")
    if params.max_mutation_probability is None:
        raise ValueError("max_mutation_probability must not be None")

    return adapt_value_by_diversity(
        value=value,
        diversity_ema=diversity_ema,
        min_threshold=params.min_diversity_threshold,
        max_threshold=params.max_diversity_threshold,
        increase_factor=params.mutation_inc_factor,
        decrease_factor=params.mutation_dec_factor,
        min_value=params.min_mutation_probability,
        max_value=params.max_mutation_probability,
    )


def adapt_crossover_probability_by_diversity(
    value: float, diversity_ema: float, params: EvoControlParams
) -> float:
    """
    Adjusts crossover probability based on diversity using threshold-based scaling.

    Increases or decreases the value depending on whether diversity falls below or above
    configured thresholds in the control parameters.
    """

    if params.min_diversity_threshold is None:
        raise ValueError("min_diversity_threshold must not be None")
    if params.max_diversity_threshold is None:
        raise ValueError("max_diversity_threshold must not be None")
    if params.crossover_inc_factor is None:
        raise ValueError("crossover_inc_factor must not be None")
    if params.crossover_dec_factor is None:
        raise ValueError("crossover_dec_factor must not be None")
    if params.min_crossover_probability is None:
        raise ValueError("min_crossover_probability must not be None")
    if params.max_crossover_probability is None:
        raise ValueError("max_crossover_probability must not be None")

    return adapt_value_by_diversity(
        value=value,
        diversity_ema=diversity_ema,
        min_threshold=params.min_diversity_threshold,
        max_threshold=params.max_diversity_threshold,
        increase_factor=params.crossover_inc_factor,
        decrease_factor=params.crossover_dec_factor,
        min_value=params.min_crossover_probability,
        max_value=params.max_crossover_probability,
    )


def exponential_decay(start: float, end: float, step: int, total_steps: int) -> float:
    """
    Applies exponential decay from `start` to `end` over `total_steps` steps.

    Args:
        start: Initial value (e.g. max strength or probability)
        end: Final value (e.g. min strength or probability)
        step: Current generation number
        total_steps: Total number of generations

    Returns:
        Decayed value at given step, clipped to [end, start]
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

    if start <= 0.0 or end <= 0.0:
        raise ValueError("start and end must be positive")

    k = np.log(start / end) / total_steps
    value = start * np.exp(-k * step)

    return float(np.clip(value, min(start, end), max(start, end)))


def exponential_mutation_strength(
    ctrl: EvoControlParams, gen: int, max_gen: int
) -> float | None:
    """Calculates exponentially decaying mutation strength over generations."""

    if ctrl.min_mutation_strength is None:
        raise ValueError("min_mutation_strength must not be None")

    if ctrl.max_mutation_strength is None:
        raise ValueError("max_mutation_strength must not be None")

    return exponential_decay(
        start=ctrl.max_mutation_strength,
        end=ctrl.min_mutation_strength,
        step=gen,
        total_steps=max_gen,
    )


def exponential_mutation_probability(
    ctrl: EvoControlParams, gen: int, max_gen: int
) -> float | None:
    """Calculates exponentially decaying mutation probability over generations."""

    if ctrl.min_mutation_probability is None:
        raise ValueError("min_mutation_probability must not be None")

    if ctrl.max_mutation_probability is None:
        raise ValueError("max_mutation_probability must not be None")

    return exponential_decay(
        start=ctrl.max_mutation_probability,
        end=ctrl.min_mutation_probability,
        step=gen,
        total_steps=max_gen,
    )


def exponential_crossover_probability(
    ctrl: EvoControlParams, gen: int, max_gen: int
) -> float | None:
    """Calculates exponentially decaying crossover probability over generations."""

    if ctrl.min_crossover_probability is None:
        raise ValueError("min_crossover_probability must not be None")

    if ctrl.max_crossover_probability is None:
        raise ValueError("max_crossover_probability must not be None")

    return exponential_decay(
        start=ctrl.max_crossover_probability,
        end=ctrl.min_crossover_probability,
        step=gen,
        total_steps=max_gen,
    )
