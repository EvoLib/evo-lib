# SPDX-License-Identifier: MIT
from typing import Any, Literal, Optional, Tuple, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from evolib.config.base_component_config import CrossoverConfig, MutationConfig
from evolib.interfaces.enums import RepresentationType


class VectorComponentConfig(BaseModel):
    """
    Configuration schema for vector-based modules (used by Vector/NetVector).

    Selected when a module has ``type: "vector"``. Defines dimensionality and (optional)
    structural interpretation, initialization, numeric bounds, and evolutionary
    operators.

    Minimal example:
        modules:
          weights:
            type: vector
            structure: flat              # "flat" | "net"
            dim: 16                      # or a list for structured cases
            initializer: normal          # name from the initializer registry
            bounds: [-1.0, 1.0]
            mutation:
              strategy: constant
              probability: 1.0
              strength: 0.05
    """

    model_config = ConfigDict(extra="forbid")

    # Fixed module type: "vector"
    type: RepresentationType = Field(
        default=RepresentationType.VECTOR,
        description='Fixed module discriminator; must be "vector" for this schema.',
    )

    # Optional structural interpretation; affects how 'dim' (and sometimes `activation`)
    # are interpreted downstream in Vector.apply_config().
    structure: Optional[Literal["flat", "net", "tensor", "blocks", "grouped"]] = Field(
        default="flat",
        description=(
            "Optional structural interpretation of the data. "
            "'flat' keeps a 1D vector; 'net' maps to a simple feed-forward layout; "
            "'tensor' treats dim as shape; 'blocks'/'grouped' flatten sums of blocks."
        ),
    )

    # Dimensionality: an integer for flat vectors, or a list for structured layouts.
    dim: Union[int, list[int]] = Field(
        ...,
        description=(
            "Vector length (int) or a list of positive integers for structured cases. "
            "Validation ensures positive sizes."
        ),
    )

    # Name of the initializer function (resolved via the initializer registry).
    initializer: str = Field(
        ...,
        description="Initializer identifier registered in "
        "evolib.initializers.registry.",
    )

    # Numeric bounds used by initialization/mutation; init_bounds may override at init.
    bounds: Tuple[float, float] = Field(
        default=(-1.0, 1.0), description="Hard clamp range for values (min, max)."
    )
    init_bounds: Optional[Tuple[float, float]] = Field(
        default=None,
        description=(
            "Initialization clamp range (min, max). Falls back to 'bounds' if omitted."
        ),
    )

    # Optional explicit shape or fixed values for the fixed initializer.
    shape: Optional[Tuple[int, ...]] = Field(
        default=None,
        description=(
            "Optional explicit shape. If provided with flat dims, Vector flattens "
            "to dim = prod(shape) and retains shape for display/reshaping."
        ),
    )
    values: Optional[list[float]] = Field(
        default=None,
        description=(
            "Explicit values for 'fixed' initializer. If 'dim' is absent it "
            "will be inferred from the length of 'values'."
        ),
    )

    # Activation (only meaningful when structure='net').
    activation: Optional[str] = Field(
        default=None,
        description=(
            "Activation for structure='net'. The Vector net mapping may use this "
            "to parameterize an internal NetVector layout."
        ),
    )

    # Evolution (mutation / crossover)
    mutation: MutationConfig = Field(
        description=(
            "Mutation configuration. By default, 'probability' is an element-wise rate "
            "(per gene) in [0,1]; operators may optionally treat it as an apply gate."
        ),
    )
    randomize_mutation_strengths: Optional[bool] = Field(
        default=False,
        description=(
            "If True, initialize per-parameter mutation strengths randomly within the "
            "configured min/max range (strategy-dependent)."
        ),
    )
    tau: Optional[float] = Field(
        default=0.0,
        description=(
            "Scale factor used by self-adaptive schemes (e.g., ADAPTIVE_*). When 0.0, "
            "some strategies will derive tau automatically."
        ),
    )
    mean: Optional[float] = Field(
        default=0.0, description="Mean for normal initializers (if applicable)."
    )
    std: Optional[float] = Field(
        default=1.0, description="Std. dev. for normal initializers (if applicable)."
    )

    crossover: Optional[CrossoverConfig] = Field(
        default=None,
        description=(
            "Crossover configuration (strategy/operator). Probability semantics are "
            "operator-dependent; see operator docs."
        ),
    )

    # Validators

    @model_validator(mode="before")
    @classmethod
    def set_dim_for_fixed_vector(cls, config: dict[str, Any]) -> dict[str, Any]:
        """
        If using 'fixed', ensure 'values' is provided and infer 'dim' if absent.

        This keeps YAML concise and catches common mistakes early.
        """
        initializer = config.get("initializer")
        values = config.get("values")

        if initializer == "fixed":
            if not values:
                raise ValueError("When using 'fixed', 'values' must be provided.")
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

    @field_validator("initializer")
    @classmethod
    def validate_initializer(cls, name: str, info: ValidationInfo) -> str:
        """Validate allowed initializer names and provide clear errors for deprecated
        names."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("initializer must be a non-empty string")

        name = name.strip()

        allowed = {"normal", "uniform", "zero", "fixed", "adaptive"}
        if name not in allowed:
            raise ValueError(
                f"Unknown initializer '{name}'. " f"Allowed: {sorted(allowed)}"
            )

        # structure-aware check (only if structure is available in the data)
        data = info.data or {}
        structure = data.get("structure") or "flat"
        if structure == "net" and name != "normal":
            raise ValueError(
                "For structure='net', initializer must be 'normal' "
                "(use initializer: normal and structure: net)."
            )

        return name
