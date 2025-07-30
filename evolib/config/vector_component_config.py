# SPDX-License-Identifier: MIT
from typing import Any, Literal, Optional, Tuple, Union

from pydantic import BaseModel, field_validator, model_validator

from evolib.config.base_component_config import CrossoverConfig, MutationConfig
from evolib.interfaces.enums import RepresentationType


class VectorComponentConfig(BaseModel):
    """
    Configuration for vector-based representations (e.g. ParaVector).

    Used with: type = "vector"

    The optional 'structure' field allows interpretation as a flat vector, network,
    tensor, or grouped blocks.

    Example:
        modules:
          weights:
            type: vector
            structure: net
            dim: [4, 8, 1]
            initializer: normal_initializer_net
            activation: relu
            bounds: [-1.0, 1.0]
    """

    type: RepresentationType = RepresentationType.VECTOR
    structure: Optional[Literal["flat", "net", "tensor", "blocks", "grouped"]] = "flat"
    dim: Union[int, list[int]]

    initializer: str

    bounds: Tuple[float, float] = (-1.0, 1.0)
    init_bounds: Optional[Tuple[float, float]] = None

    shape: Optional[Tuple[int, ...]] = None
    values: Optional[list[float]] = None
    activation: Optional[str] = None  # only used for structure = "net"

    # Mutation configuration
    mutation: Optional[MutationConfig] = None
    randomize_mutation_strengths: Optional[bool] = False
    tau: Optional[float] = 0.0
    mean: Optional[float] = 0.0
    std: Optional[float] = 1.0

    # Crossover configuration
    crossover: Optional[CrossoverConfig] = None

    @model_validator(mode="before")
    @classmethod
    def set_dim_for_fixed_vector(cls, config: dict[str, Any]) -> dict[str, Any]:
        initializer = config.get("initializer")
        values = config.get("values")

        if initializer == "fixed_vector":
            if not values:
                raise ValueError(
                    "When using 'fixed_initializer', 'values' must be provided."
                )
            if "dim" not in config:
                config["dim"] = len(values)
        return config

    @field_validator("bounds", "init_bounds")
    @classmethod
    def check_bounds(cls, bounds: Tuple[float, float]) -> Tuple[float, float]:
        low, high = bounds
        if low > high:
            raise ValueError("Bounds must be specified as (min, max) with min <= max")
        return bounds

    @field_validator("dim")
    @classmethod
    def validate_dim(cls, dim: Union[int, list[int]]) -> Union[int, list[int]]:
        if isinstance(dim, int):
            if dim <= 0:
                raise ValueError("dim must be a positive integer")
        elif isinstance(dim, list):
            if not dim or not all(isinstance(d, int) and d > 0 for d in dim):
                raise ValueError("dim must be a non-empty list of positive integers")
        else:
            raise TypeError("dim must be an int or list of ints")
        return dim
