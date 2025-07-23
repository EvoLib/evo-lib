# SPDX-License-Identifier: MIT
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, model_validator

from evolib.interfaces.enums import (
    CrossoverOperator,
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
    operator: Optional[CrossoverOperator] = None
    probability: Optional[float] = None
    init_probability: Optional[float] = None
    min_probability: Optional[float] = None
    max_probability: Optional[float] = None
    increase_factor: Optional[float] = None
    decrease_factor: Optional[float] = None
    # Parameters for specific operators
    alpha: Optional[float] = None  # for BLX
    eta: Optional[float] = None  # for SBX
    blend_range: Optional[float] = None  # for Intermediate


class ComponentConfig(BaseModel):
    type: RepresentationType = RepresentationType.VECTOR
    dim: Optional[int] = None
    structure: Optional[list[int]] = None
    bounds: Tuple[float, float] = (-1.0, 1.0)
    initializer: str
    init_bounds: Optional[Tuple[float, float]] = None
    randomize_mutation_strengths: Optional[bool] = False
    tau: Optional[float] = 0.0
    mean: Optional[float] = 0.0
    std: Optional[float] = 0.0
    values: Optional[List[float]] = None

    mutation: Optional[MutationConfig] = None
    crossover: Optional[CrossoverConfig] = None

    @model_validator(mode="before")
    @classmethod
    def validate_initializer_and_dim(cls, data: dict[str, Any]) -> dict[str, Any]:
        initializer = data.get("initializer")
        dim = data.get("dim")
        values = data.get("values")
        structure = data.get("structure")

        # Fall 1: fixed_initializer → values müssen gesetzt sein
        if initializer == "fixed_initializer":
            if not values:
                raise ValueError(
                    "When using 'fixed_initializer', 'values' must be provided."
                )
            if dim is None:
                data["dim"] = len(values)

        # Fall 2: andere Initializer
        else:
            if dim is None and not structure:
                raise ValueError(
                    "Field 'dim' is required unless 'structure' is provided."
                )
            if values is not None:
                raise ValueError(
                    "Field 'values' must not be set unless initializer "
                    "is 'fixed_initializer'."
                )

        # Fall 3: structure → dim ableiten
        if "dim" not in data and structure:
            if isinstance(structure, list) and len(structure) >= 2:
                data["dim"] = sum(i * j for i, j in zip(structure, structure[1:]))

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
    max_indiv_age: int = 0
    num_elites: int

    modules: Dict[str, ComponentConfig]

    evolution: Optional[EvolutionConfig] = None
    selection: Optional[SelectionConfig] = None
    replacement: Optional[ReplacementConfig] = None
