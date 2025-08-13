# SPDX-License-Identifier: MIT

from evolib.config.base_component_config import CrossoverConfig, MutationConfig
from evolib.interfaces.enums import CrossoverStrategy, MutationStrategy
from evolib.representation.evo_params import EvoControlParams


def apply_mutation_config(ep: EvoControlParams, m: MutationConfig) -> None:
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
    if c is None:
        ep.crossover_strategy = CrossoverStrategy.NONE
        return
    ep.crossover_strategy = c.strategy
    ep.crossover_probability = c.probability or c.init_probability
    ep.min_crossover_probability = c.min_probability
    ep.max_crossover_probability = c.max_probability
    ep.crossover_inc_factor = c.increase_factor
    ep.crossover_dec_factor = c.decrease_factor
