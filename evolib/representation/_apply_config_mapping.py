# SPDX-License-Identifier: MIT
"""
Mapping functions to translate high-level MutationConfig and CrossoverConfig into
concrete EvoControlParams fields.

This ensures that configuration objects (parsed via Pydantic) are consistently applied
to EvoControlParams, which store runtime control parameters used during evolutionary
runs.
"""

from evolib.config.base_component_config import (
    CrossoverConfig,
    CrossoverOperator,
    MutationConfig,
)
from evolib.interfaces.enums import CrossoverStrategy, MutationStrategy
from evolib.operators.crossover import (
    crossover_arithmetic,
    crossover_blend_alpha,
    crossover_intermediate,
    crossover_simulated_binary,
)
from evolib.representation.evo_params import EvoControlParams


def apply_mutation_config(ep: EvoControlParams, m: MutationConfig) -> None:
    """
    Map a MutationConfig into EvoControlParams.

    Args:
        ep (EvoControlParams): Target container that will be updated in-place.
        m (MutationConfig): Parsed config object defining mutation behavior.

    The mapping depends on the selected MutationStrategy:
        - CONSTANT: fixed probability and strength
        - EXPONENTIAL_DECAY: min/max ranges for probability and strength
        - ADAPTIVE_GLOBAL: initial values plus min/max ranges and diversity factors
        - ADAPTIVE_INDIVIDUAL: per-individual bounds for adaptive mutation strength
        - ADAPTIVE_PER_PARAMETER: per-parameter bounds for adaptive mutation strength

    Raises:
        ValueError: If the strategy is unknown.
    """
    ep.mutation_strategy = m.strategy

    if m.strategy == MutationStrategy.CONSTANT:
        ep.mutation_probability = m.probability
        ep.mutation_strength = m.strength

    elif m.strategy == MutationStrategy.EXPONENTIAL_DECAY:
        ep.min_mutation_probability = m.min_probability
        ep.max_mutation_probability = m.max_probability
        ep.min_mutation_strength = m.min_strength
        ep.max_mutation_strength = m.max_strength

    elif m.strategy == MutationStrategy.ADAPTIVE_GLOBAL:
        ep.mutation_probability = m.init_probability
        ep.mutation_strength = m.init_strength
        ep.min_mutation_probability = m.min_probability
        ep.max_mutation_probability = m.max_probability
        ep.min_mutation_strength = m.min_strength
        ep.max_mutation_strength = m.max_strength
        ep.min_diversity_threshold = m.min_diversity_threshold
        ep.max_diversity_threshold = m.max_diversity_threshold
        ep.mutation_inc_factor = m.increase_factor
        ep.mutation_dec_factor = m.decrease_factor

    elif m.strategy == MutationStrategy.ADAPTIVE_INDIVIDUAL:
        ep.min_mutation_strength = m.min_strength
        ep.max_mutation_strength = m.max_strength

    elif m.strategy == MutationStrategy.ADAPTIVE_PER_PARAMETER:
        ep.min_mutation_strength = m.min_strength
        ep.max_mutation_strength = m.max_strength

    else:
        raise ValueError(f"Unknown mutation strategy: {m.strategy}")


def apply_crossover_config(ep: EvoControlParams, c: CrossoverConfig | None) -> None:
    """
    Map a CrossoverConfig into EvoControlParams.

    Args:
        ep (EvoControlParams): Target container updated in-place.
        c (CrossoverConfig | None): Parsed config object. If None,
            disables crossover (CrossoverStrategy.NONE).

    Mapping rules:
        - If config is None -> crossover disabled.
        - Otherwise -> map strategy, probability and adaptive factors.
    """
    if c is None:
        ep.crossover_strategy = CrossoverStrategy.NONE
        ep._crossover_fn = None
        return

    ep.crossover_strategy = c.strategy

    if c.probability is not None:
        ep.crossover_probability = c.probability
    elif c.init_probability is not None:
        ep.crossover_probability = c.init_probability

    ep.min_crossover_probability = c.min_probability
    ep.max_crossover_probability = c.max_probability
    ep.crossover_inc_factor = c.increase_factor
    ep.crossover_dec_factor = c.decrease_factor

    op = c.operator
    if op == CrossoverOperator.BLX:
        alpha = c.alpha or 0.5
        ep._crossover_fn = lambda a, b: crossover_blend_alpha(a, b, alpha)
    elif op == CrossoverOperator.ARITHMETIC:
        ep._crossover_fn = crossover_arithmetic
    elif op == CrossoverOperator.SBX:
        eta = c.eta or 15.0
        ep._crossover_fn = lambda a, b: crossover_simulated_binary(a, b, eta)
    elif op == CrossoverOperator.INTERMEDIATE:
        blend = c.blend_range or 0.25
        ep._crossover_fn = lambda a, b: crossover_intermediate(a, b, blend)
    else:
        ep._crossover_fn = None
