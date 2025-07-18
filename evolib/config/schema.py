# SPDX-License-Identifier: MIT
"""Defines the pydantic schema for YAML-based configuration files in EvoLib."""

from typing import Any, List, Optional, Tuple

from pydantic import BaseModel, Field, model_validator

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
    init_probability: Optional[float] = None
    probability: Optional[float] = None
    init_strength: Optional[float] = None
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
    init_probability: Optional[float] = None
    min_probability: Optional[float] = None
    max_probability: Optional[float] = None
    increase_factor: Optional[float] = None
    decrease_factor: Optional[float] = None


class RepresentationConfig(BaseModel):
    type: RepresentationType
    dim: Optional[int] = None
    bounds: Tuple[float, float]
    initializer: str
    values: Optional[List[float]] = None  # Nur für fixed
    randomize_mutation_strengths: Optional[bool] = False
    init_bounds: Optional[Tuple[float, float]] = None
    tau: Optional[float] = 0.0
    mean: Optional[float] = 0.0
    std: Optional[float] = 0.0

    @model_validator(mode="before")
    @classmethod
    def check_fixed_initializer(cls, data: dict[str, Any]) -> dict[str, Any]:
        initializer = data.get("initializer")
        dim = data.get("dim")
        values = data.get("values")

        if initializer == "fixed_initializer":
            if not values:
                raise ValueError(
                    "When using 'fixed' initializer, " "'values' must be provided."
                )
            if dim is None:
                data["dim"] = len(values)
        else:
            if dim is None:
                raise ValueError(
                    "Field 'dim' is required for initializers other " "than 'fixed'."
                )
            if values is not None:
                raise ValueError(
                    "Field 'values' must not be set unless initializer " "is 'fixed'."
                )

        return data


class EvolutionConfig(BaseModel):
    strategy: EvolutionStrategy


class SelectionConfig(BaseModel):
    strategy: SelectionStrategy
    num_parents: Optional[int] = None
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
