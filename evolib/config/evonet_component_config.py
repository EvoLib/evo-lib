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

from evonet.activation import ACTIVATIONS
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
    validator,
)
from pydantic_core import core_schema

from evolib.config.base_component_config import (
    CrossoverConfig,
    EvoNetMutationConfig,
    EvoNetNeuronDynamicsConfig,
    StructuralMutationConfig,
)

Bounds = Tuple[float, float]


class WeightsConfig(BaseModel):
    # None means: no parameter-level initialization requested (preset may initialize).
    initializer: Optional[str] = None  # normal | uniform | zero | None
    std: Optional[float] = None
    bounds: Bounds = (-1.0, 1.0)  # mutation/search bounds
    init_bounds: Optional[Bounds] = None  # init-time clipping bounds

    @model_validator(mode="after")
    def _validate(self) -> "WeightsConfig":
        # Validate bounds
        lo, hi = self.bounds
        if lo >= hi:
            raise ValueError("weights.bounds must satisfy lower < upper")

        if self.init_bounds is not None:
            ilo, ihi = self.init_bounds
            if ilo >= ihi:
                raise ValueError("weights.init_bounds must satisfy lower < upper")
            if not (lo <= ilo and ihi <= hi):
                raise ValueError("weights.init_bounds must lie within weights.bounds")

        # Validate initializer-specific fields
        if self.initializer is None:
            # Preset-based initialization
            if self.std is not None:
                raise ValueError(
                    "weights.std is not allowed when " "weights.initializer is not set"
                )
            return self

        if self.initializer == "normal":
            if self.std is None or self.std <= 0:
                raise ValueError(
                    "weights.std must be set and > 0 for initializer=normal"
                )

        elif self.initializer in ("uniform", "zero"):
            if self.std is not None:
                raise ValueError("weights.std is only allowed for initializer=normal")

        else:
            raise ValueError(f"Unknown weights initializer: {self.initializer}")

        return self


class BiasConfig(BaseModel):
    # None means: no parameter-level initialization requested (preset may initialize).
    initializer: Optional[str] = None  # fixed | normal | uniform | zero | None
    std: Optional[float] = None
    value: Optional[float] = None
    bounds: Bounds = (-0.5, 0.5)  # mutation/search bounds
    init_bounds: Optional[Bounds] = None  # init-time clipping bounds (optional)

    @model_validator(mode="after")
    def _validate(self) -> "BiasConfig":
        # Validate bounds
        lo, hi = self.bounds
        if lo >= hi:
            raise ValueError("bias.bounds must satisfy lower < upper")

        if self.init_bounds is not None:
            ilo, ihi = self.init_bounds
            if ilo >= ihi:
                raise ValueError("bias.init_bounds must satisfy lower < upper")
            if not (lo <= ilo and ihi <= hi):
                raise ValueError("bias.init_bounds must lie within bias.bounds")

        # Validate initializer-specific fields
        if self.initializer is None:
            # Preset-based initialization
            if self.std is not None:
                raise ValueError(
                    "bias.std is not allowed when bias.initializer is not set"
                )
            if self.value is not None:
                raise ValueError(
                    "bias.value is not allowed when bias.initializer is not set"
                )
            return self

        if self.initializer == "fixed":
            if self.value is None:
                raise ValueError("bias.value is required for initializer=fixed")
            if self.std is not None:
                raise ValueError("bias.std is not allowed for initializer=fixed")

        elif self.initializer == "normal":
            if self.std is None or self.std <= 0:
                raise ValueError("bias.std must be set and > 0 for initializer=normal")
            if self.value is not None:
                raise ValueError("bias.value is only allowed for initializer=fixed")

        elif self.initializer in ("uniform", "zero"):
            if self.std is not None:
                raise ValueError("bias.std is only allowed for initializer=normal")
            if self.value is not None:
                raise ValueError("bias.value is only allowed for initializer=fixed")

        else:
            raise ValueError(f"Unknown bias initializer: {self.initializer}")

        return self


class DelayConfig(BaseModel):
    """Delay initialization for recurrent connections."""

    initializer: Literal["fixed", "uniform"] = Field(
        default="fixed",
        description="Delay initializer for recurrent connections.",
    )

    value: Optional[int] = Field(
        default=None,
        ge=1,
        description="Fixed delay value (required for initializer=fixed).",
    )

    bounds: Optional[Tuple[int, int]] = Field(
        default=None,
        description="Inclusive [min, max] delay bounds (required for "
        "initializer=uniform).",
    )

    @model_validator(mode="after")
    def _validate(self) -> "DelayConfig":
        if self.initializer == "fixed":
            if self.value is None:
                raise ValueError("delay.value is required for initializer=fixed")
        elif self.initializer == "uniform":
            if self.bounds is None:
                raise ValueError("delay.bounds is required for initializer=uniform")
            lo, hi = self.bounds
            if lo < 1 or hi < 1:
                raise ValueError("delay bounds must be >= 1 (recurrent-only)")
            if hi < lo:
                raise ValueError("delay.bounds[1] must be >= delay.bounds[0]")
        return self


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

            weights:
              initializer: normal
              std: 0.5
              bounds: [-1.0, 1.0]

            bias:
              initializer: uniform
              bounds: [-0.5, 0.5]

            mutation:
              strategy: constant
              probability: 0.8
              strength: 0.05
    """

    model_config = ConfigDict(extra="forbid")

    # Module type is fixed to "evonet"
    type: Literal["evonet"] = "evonet"

    # Layer structure: list of neuron counts per layer [input, hidden..., output]
    dim: list[int]

    # Either a single activation function or one per layer
    activation: Union[str, list[str]] = "tanh"

    # Whitelist for random activation selection
    activations_allowed: Optional[list[str]] = Field(
        default=None,
        description="Whitelist of activation names applied to "
        "neurons in hidden layers.",
    )

    # Recurrent connections
    recurrent: Optional[Literal["none", "direct", "local", "all"]] = "none"

    # Name of the initializer function (resolved via initializer registry)
    initializer: str = Field(
        default="default_evonet", description="Name of the initializer to use"
    )

    # Connection topology for initialization
    connection_scope: Literal["adjacent", "crosslayer"] = Field(
        default="adjacent",
        description=(
            "Defines how layers are connected during initialization. "
            "'adjacent' connects only consecutive layers, while 'crosslayer' "
            "connects all earlier layers to all later layers."
        ),
    )

    connection_density: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of possible connections actually created during initialization. "
            "1.0 = fully connected, <1.0 = sparse."
        ),
    )

    # Numeric bounds for values; used by initialization and mutation
    weights: WeightsConfig = Field(default_factory=WeightsConfig)
    bias: BiasConfig = Field(default_factory=BiasConfig)

    # Optional delay initialization for recurrent connections
    delay: Optional[DelayConfig] = None

    # Neuron Dynamics
    neuron_dynamics: Optional[list[EvoNetNeuronDynamicsConfig]] = None

    # Evolutionary operators
    mutation: Optional[EvoNetMutationConfig] = None
    crossover: Optional[CrossoverConfig] = None
    structural: Optional[StructuralMutationConfig] = None

    # Validators
    @field_validator("initializer")
    @classmethod
    def validate_initializer(cls, name: str) -> str:
        """
        Validate that the initializer is one of the allowed topology presets.

        Parameter initialization is handled exclusively via weights/bias/delay blocks.
        """
        allowed = {
            "default_evonet",
            "unconnected_evonet",
            "identity_evonet",
        }

        if name not in allowed:
            raise ValueError(
                f"Unknown EvoNet initializer '{name}'. "
                f"Allowed values: {sorted(allowed)}. "
                "Parameter initialization is configured via "
                "'weights', 'bias', and 'delay'."
            )

        return name

    @field_validator("neuron_dynamics")
    @classmethod
    def validate_neuron_dynamics_length(
        cls,
        nd: Optional[list[EvoNetNeuronDynamicsConfig]],
        info: core_schema.FieldValidationInfo,
    ) -> Optional[list[EvoNetNeuronDynamicsConfig]]:
        if nd is None:
            return None

        dim = info.data.get("dim")
        if dim is not None and len(nd) != len(dim):
            raise ValueError("Length of 'neuron_dynamics' must match 'dim'")

        return nd

    @field_validator("dim")
    @classmethod
    def check_valid_layer_structure(cls, dim: list[int]) -> list[int]:
        """Ensure that `dim` has at least input/output layer and all values > 0."""
        if len(dim) < 2:
            raise ValueError("dim must contain at least input and output layer")
        if not all(isinstance(x, int) and x >= 0 for x in dim):
            raise ValueError("All layer sizes in dim must be non-negative integers")
        return dim

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

    @validator("activations_allowed", each_item=True)
    def validate_activation_name(cls, act_name: str) -> str:
        """Ensure only valid activation function names are allowed."""
        if act_name not in ACTIVATIONS:
            raise ValueError(
                f"Invalid activation function '{act_name}'. "
                f"Valid options are: {list(ACTIVATIONS.keys())}"
            )
        return act_name

    @validator("recurrent")
    def validate_recurrent(cls, recurrent: Optional[str]) -> str:
        """Ensure recurrent preset is valid and normalized."""
        if recurrent is None:
            return "none"
        allowed = {"none", "direct", "local", "all"}
        if recurrent not in allowed:
            raise ValueError(
                f"Invalid recurrent preset '{recurrent}'. "
                f"Valid options are: {sorted(allowed)}"
            )
        return recurrent

    @field_validator("connection_scope")
    @classmethod
    def validate_connection_scope(cls, scope: str) -> str:
        """Ensure connection_scope is one of the supported options."""
        allowed = {"adjacent", "crosslayer"}
        if scope not in allowed:
            raise ValueError(
                f"Invalid connection_scope '{scope}'. "
                f"Valid options are: {sorted(allowed)}"
            )
        return scope

    @field_validator("connection_density")
    @classmethod
    def validate_connection_density(cls, density: float) -> float:
        """Ensure connection_density is within [0, 1]."""
        if not (0.0 <= density <= 1.0):
            raise ValueError(f"connection_density must be in [0, 1], got {density}.")
        return density
