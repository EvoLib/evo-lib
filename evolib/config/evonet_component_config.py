# SPDX-License-Identifier: MIT
"""
EvoNetComponentConfig defines the configuration schema for structured evolutionary
neural networks (EvoNet).

This config class is selected when a module has `type: "evonet"`. It validates layer
dimensions, activation functions, weight/bias bounds, and optional mutation/crossover
strategies.

After parsing, raw dicts are converted into this strongly typed Pydantic model during
config resolution.
"""

from typing import Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator
from pydantic_core import core_schema

from evolib.config.base_component_config import (
    CrossoverConfig,
    EvoNetMutationConfig,
    StructuralMutationConfig,
)


class EvoNetComponentConfig(BaseModel):
    """
    Configuration schema for EvoNet modules (used by EvoNet).

    This config is selected when a module has ``type: "evonet"`` and defines
    the structure, initialization, and evolutionary operators for EvoNet-based
    neural networks.

    Minimal example:
        modules:
          brain:
            type: evonet
            dim: [4, 6, 2]                       # input, hidden, output
            activation: [linear, relu, sigmoid]  # single activation or list per layer
            initializer: normal_evonet           # weight/bias initializer
            weight_bounds: [-1.0, 1.0]
            bias_bounds: [-0.5, 0.5]
            mutation:
              strategy: constant
              probability: 0.8
              strength: 0.05
    """

    # Module type is fixed to "evonet"
    type: Literal["evonet"] = "evonet"

    # Layer structure: list of neuron counts per layer [input, hidden..., output]
    dim: list[int]

    # Either a single activation function or one per layer
    activation: Union[str, list[str]] = "tanh"

    # Name of the initializer function (resolved via initializer registry)
    initializer: str = Field(..., description="Name of the initializer to use")

    # Numeric bounds for values; used by initialization and mutation
    weight_bounds: Tuple[float, float] = (-1.0, 1.0)
    bias_bounds: Tuple[float, float] = (-0.5, 0.5)

    # Evolutionary operators
    mutation: Optional[EvoNetMutationConfig] = None
    crossover: Optional[CrossoverConfig] = None
    structural: Optional[StructuralMutationConfig] = None

    # Validators

    @field_validator("dim")
    @classmethod
    def check_valid_layer_structure(cls, dim: list[int]) -> list[int]:
        """Ensure that `dim` has at least input/output layer and all values > 0."""
        if len(dim) < 2:
            raise ValueError("dim must contain at least input and output layer")
        if not all(isinstance(x, int) and x >= 0 for x in dim):
            raise ValueError("All layer sizes in dim must be non-negative integers")
        return dim

    @field_validator("weight_bounds", "bias_bounds")
    @classmethod
    def check_bounds(cls, bounds: Tuple[float, float]) -> Tuple[float, float]:
        """Validate that bounds are well-formed (min < max)."""
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
        """If a list of activations is given, ensure its length matches the number of
        layers."""
        dim = info.data.get("dim")
        if isinstance(act, list) and dim and len(act) != len(dim):
            raise ValueError("Length of 'activation' list must match 'dim'")
        return act
