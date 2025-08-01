# SPDX-License-Identifier: MIT
"""
EvoNetComponentConfig defines the configuration schema for structured evolutionary
neural networks.

It is used with type = "evonet" and supports definition of:
    - Layer structure (dim)
    - Activation function
    - Initialization bounds for weights and biases
    - Optional mutation/crossover strategies
"""

from typing import Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator
from pydantic_core import core_schema

from evolib.config.base_component_config import CrossoverConfig, MutationConfig


class EvoNetComponentConfig(BaseModel):
    """
    Configuration for structured EvoNet-like networks (e.g. ParaEvoNet).

    Used with: type = "evonet"

    Example:
        modules:
          brain:
            type: evonet
            dim: [4, 6, 2]
            activation: "relu"
            initializer: "normal_evonet"
            weight_bounds: [-1.0, 1.0]
            bias_bounds: [-0.5, 0.5]
    """

    type: Literal["evonet"] = "evonet"
    dim: list[int]

    activation: Union[str, list[str]] = "tanh"
    initializer: str = Field(..., description="Name of the initializer to use")

    weight_bounds: Tuple[float, float] = (-1.0, 1.0)
    bias_bounds: Tuple[float, float] = (-0.5, 0.5)

    mutation: Optional[MutationConfig] = None
    crossover: Optional[CrossoverConfig] = None

    @field_validator("dim")
    @classmethod
    def check_valid_layer_structure(cls, dim: list[int]) -> list[int]:
        if len(dim) < 2:
            raise ValueError("dim must contain at least input and output layer")
        if not all(isinstance(x, int) and x > 0 for x in dim):
            raise ValueError("All layer sizes in dim must be positive integers")
        return dim

    @field_validator("weight_bounds", "bias_bounds")
    @classmethod
    def check_bounds(cls, bounds: Tuple[float, float]) -> Tuple[float, float]:
        low, high = bounds
        if low >= high:
            raise ValueError("Bounds must be specified as (min, max) with min < max")
        return bounds

    @field_validator("activation")
    @classmethod
    def validate_activation_length(
        cls,
        act: Union[str, list[str]],
        info: core_schema.FieldValidationInfo,
    ) -> Union[str, list[str]]:

        dim = info.data.get("dim")
        if isinstance(act, list) and dim and len(act) != len(dim):
            raise ValueError("Length of 'activation' list must match 'dim'")
        return act
