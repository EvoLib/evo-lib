# SPDX-License-Identifier: MIT
"""Defines the pydantic schema for YAML-based configuration files in EvoLib."""

from typing import Optional, Tuple

from pydantic import BaseModel, Field

from evolib.interfaces.enums import (
    CrossoverStrategy,
    EvolutionStrategy,
    MutationStrategy,
    ReplacementStrategy,
    RepresentationType,
    SelectionStrategy,
)


class MutationConfig(BaseModel):
    strategy: MutationStrategy
    strength: Optional[float] = None
    probability: Optional[float] = None
    min_strength: Optional[float] = None
    max_strength: Optional[float] = None
    min_probability: Optional[float] = None
    max_probability: Optional[float] = None
    increase_factor: Optional[float] = None
    decrease_factor: Optional[float] = None
    min_diversity_threshold: Optional[float] = None
    max_diversity_threshold: Optional[float] = None


class CrossoverConfig(BaseModel):
    strategy: CrossoverStrategy
    probability: Optional[float] = None
    min_probability: Optional[float] = None
    max_probability: Optional[float] = None
    increase_factor: Optional[float] = None
    decrease_factor: Optional[float] = None


class RepresentationConfig(BaseModel):
    type: RepresentationType
    dim: int
    bounds: Tuple[float, float]
    initializer: str
    randomize_mutation_strengths: Optional[bool] = False
    init_bounds: Optional[Tuple[float, float]] = None


class EvolutionConfig(BaseModel):
    strategy: EvolutionStrategy


class SelectionConfig(BaseModel):
    strategy: SelectionStrategy
    tournament_size: Optional[int] = None
    exp_base: Optional[float] = None
    fitness_maximization: Optional[bool] = False


class ReplacementConfig(BaseModel):
    strategy: ReplacementStrategy = Field(
        ..., description="Replacement strategy to use for survivor selection."
    )

    num_replace: Optional[int] = Field(
        default=None,
        description="Number of individuals to replace (only used by steady_state).",
    )

    temperature: Optional[float] = Field(
        default=None, description="Temperature for stochastic (softmax) replacement."
    )


class FullConfig(BaseModel):
    parent_pool_size: int
    offspring_pool_size: int
    max_generations: int
    max_indiv_age: int
    num_elites: int
    representation: RepresentationConfig
    mutation: MutationConfig
    crossover: Optional[CrossoverConfig] = None
    evolution: Optional[EvolutionConfig] = None
    selection: Optional[SelectionConfig] = None
    replacement: Optional[ReplacementConfig] = None
