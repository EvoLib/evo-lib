# SPDX-License-Identifier: MIT
from typing import Any, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from evolib.config.base_component_config import CrossoverConfig, MutationConfig
from evolib.interfaces.enums import RepresentationType


class VectorComponentConfig(BaseModel):
    """Configuration schema for flat vector modules."""

    model_config = ConfigDict(extra="forbid")

    type: RepresentationType = Field(
        default=RepresentationType.VECTOR,
        description='Fixed module discriminator; must be "vector" for this schema.',
    )

    dim: int = Field(
        ...,
        gt=0,
        description="Number of scalar parameters in the vector.",
    )

    initializer: str = Field(
        ...,
        description="Initializer identifier registered in "
        "evolib.initializers.registry.",
    )

    bounds: Tuple[float, float] = Field(
        default=(-1.0, 1.0), description="Hard clamp range for values (min, max)."
    )
    init_bounds: Optional[Tuple[float, float]] = Field(
        default=None,
        description=(
            "Initialization clamp range (min, max). Falls back to 'bounds' if omitted."
        ),
    )

    values: Optional[list[float]] = Field(
        default=None,
        description=(
            "Explicit values for 'fixed' initializer. If 'dim' is absent it "
            "will be inferred from the length of 'values'."
        ),
    )

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

    @field_validator("initializer")
    @classmethod
    def validate_initializer(cls, name: str) -> str:
        """Validate the initializer name."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("initializer must be a non-empty string")

        name = name.strip()
        allowed = {"normal", "uniform", "zero", "fixed", "adaptive"}
        if name not in allowed:
            raise ValueError(
                f"Unknown initializer '{name}'. Allowed: {sorted(allowed)}"
            )
        return name
