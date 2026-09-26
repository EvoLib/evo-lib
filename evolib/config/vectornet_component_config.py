# SPDX-License-Identifier: MIT
"""Configuration schema for fixed-topology VectorNet modules."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from evolib.config.base_component_config import CrossoverConfig, MutationConfig
from evolib.interfaces.enums import RepresentationType


class VectorNetComponentConfig(BaseModel):
    """Configuration for a fixed-topology neural network backed by a vector."""

    model_config = ConfigDict(extra="forbid")

    type: RepresentationType = Field(
        default=RepresentationType.VECTORNET,
        description='Fixed module discriminator; must be "vectornet" for this schema.',
    )

    dim: list[int] = Field(
        ...,
        description="Positive layer sizes including input and output layers.",
    )
    activation: Literal["tanh", "relu", "linear"] = Field(
        default="tanh",
        description="Activation applied after each hidden layer.",
    )

    initializer: Literal["normal"] = Field(
        default="normal",
        description="Parameter initializer. VectorNet currently supports normal only.",
    )
    mean: float = Field(default=0.0, description="Mean of the normal initializer.")
    std: float = Field(
        default=1.0,
        ge=0.0,
        description="Standard deviation of the normal initializer.",
    )

    bounds: tuple[float, float] = Field(
        default=(-1.0, 1.0),
        description="Hard clamp range for parameters (min, max).",
    )
    init_bounds: tuple[float, float] | None = Field(
        default=None,
        description=(
            "Initialization clamp range (min, max). Falls back to bounds if omitted."
        ),
    )

    mutation: MutationConfig = Field(description="Mutation configuration.")
    tau: float = Field(
        default=0.0,
        description="Scale factor used by self-adaptive mutation strategies.",
    )
    crossover: CrossoverConfig | None = Field(
        default=None,
        description="Optional crossover configuration.",
    )

    @field_validator("dim")
    @classmethod
    def validate_dim(cls, dim: list[int]) -> list[int]:
        """Require at least input and output layers with positive sizes."""
        if len(dim) < 2 or not all(isinstance(size, int) and size > 0 for size in dim):
            raise ValueError("dim must contain at least two positive layer sizes")
        return dim

    @field_validator("bounds", "init_bounds")
    @classmethod
    def validate_bounds(
        cls, bounds: tuple[float, float] | None
    ) -> tuple[float, float] | None:
        """Ensure numeric bounds are ordered from minimum to maximum."""
        if bounds is None:
            return None
        low, high = bounds
        if low > high:
            raise ValueError("Bounds must be specified as (min, max) with min <= max")
        return bounds
