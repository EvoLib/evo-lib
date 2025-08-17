# SPDX-License-Identifier: MIT
from typing import Any, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from evolib.config.base_component_config import CrossoverConfig, MutationConfig
from evolib.interfaces.enums import RepresentationType


class VectorComponentConfig(BaseModel):
    """
    Configuration schema for vector-based modules (used by ParaVector/NetVector).

    This config is selected when a module has type: "vector". It defines the
    dimensionality, optional structural interpretation (e.g., flat vector vs. net),
    initialization, numeric bounds, and optional evolutionary operators.

    Minimal example:
        modules:
          weights:
            type: vector
            structure: flat            # "flat" | "net"
            dim: 16                    # or a list for structured cases
            initializer: random_vector # name from the initializer registry
            bounds: [-1.0, 1.0]
            mutation:
              strategy: constant
              probability: 1.0
              strength: 0.05
    """

    # Fixed module type: "vector"
    type: RepresentationType = RepresentationType.VECTOR

    # Optional structural interpretation of the data; defaults to a flat vector.
    # "net" and other structures may influence how 'dim' and 'activation'
    # are interpreted.
    structure: Optional[Literal["flat", "net", "tensor", "blocks", "grouped"]] = "flat"

    # Dimensionality: integer for flat vectors, or a list for structured layouts
    dim: Union[int, list[int]]

    # Name of the initializer function (resolved via initializer registry)
    initializer: str = Field(..., description="Name of the initializer to use")

    # Numeric bounds for values; used by initialization and mutation
    bounds: Tuple[float, float] = (-1.0, 1.0)

    # Optional separate init-bounds (fallbacks to 'bounds' if not set)
    init_bounds: Optional[Tuple[float, float]] = None

    # Optional explicit shape and values for fixed initializers
    shape: Optional[Tuple[int, ...]] = None
    values: Optional[list[float]] = None

    # Optional activation hint (only meaningful for structure="net")
    activation: Optional[str] = None

    # Mutation / evolution configuration

    mutation: Optional[MutationConfig] = None
    randomize_mutation_strengths: Optional[bool] = False
    tau: Optional[float] = 0.0
    mean: Optional[float] = 0.0
    std: Optional[float] = 1.0

    # Crossover configuration

    crossover: Optional[CrossoverConfig] = None

    # Validators

    @model_validator(mode="before")
    @classmethod
    def set_dim_for_fixed_vector(cls, config: dict[str, Any]) -> dict[str, Any]:
        """If using 'fixed_vector', ensure 'values' is provided and infer 'dim' if
        absent."""
        initializer = config.get("initializer")
        values = config.get("values")

        if initializer == "fixed_vector":
            if not values:
                raise ValueError(
                    "When using 'fixed_vector', 'values' must be provided."
                )
            if "dim" not in config:
                config["dim"] = len(values)
        return config

    @field_validator("bounds", "init_bounds")
    @classmethod
    def check_bounds(cls, bounds: Tuple[float, float]) -> Tuple[float, float]:
        """Validate that bounds are well-formed (min <= max)."""
        low, high = bounds
        if low > high:
            raise ValueError("Bounds must be specified as (min, max) with min <= max")
        return bounds

    @field_validator("dim")
    @classmethod
    def validate_dim(cls, dim: Union[int, list[int]]) -> Union[int, list[int]]:
        """Require a positive integer or a non-empty list of positive integers."""
        if isinstance(dim, int):
            if dim <= 0:
                raise ValueError("dim must be a positive integer")
        elif isinstance(dim, list):
            if not dim or not all(isinstance(d, int) and d > 0 for d in dim):
                raise ValueError("dim must be a non-empty list of positive integers")
        else:
            raise TypeError("dim must be an int or list of ints")
        return dim
