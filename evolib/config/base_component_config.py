# SPDX-License-Identifier: MIT
"""
Shared configuration blocks for mutation and crossover used across multiple
ComponentConfig classes (e.g. VectorComponentConfig, EvoNetComponentConfig).

The classes here are intentionally small and reusable: they describe *what*
should be configured, not *how* it is executed. Any runtime behavior belongs
into the respective Para* representations and operator modules.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from evolib.interfaces.enums import (
    CrossoverOperator,
    CrossoverStrategy,
    MutationStrategy,
)


class MutationConfig(BaseModel):
    """
    Configuration block for mutation strategies.

    Supported strategies (see MutationStrategy enum):
        - CONSTANT
        - EXPONENTIAL_DECAY
        - ADAPTIVE_GLOBAL
        - ADAPTIVE_INDIVIDUAL
        - ADAPTIVE_PER_PARAMETER

    Which fields are relevant depends on the selected strategy:

    CONSTANT
        - strength (required)
        - probability (optional; default behavior handled downstream)

    EXPONENTIAL_DECAY
        - init_strength (required)
        - init_probability (optional)

    ADAPTIVE_GLOBAL
        - strength (required as starting point; mapped to runtime state)
        - probability (required as starting point)
        - increase_factor / decrease_factor (optional)
        - min_diversity_threshold / max_diversity_threshold (optional)
        - min_strength / max_strength (optional clamp)
        - min_probability / max_probability (optional clamp)

    ADAPTIVE_INDIVIDUAL / ADAPTIVE_PER_PARAMETER
        - min_strength, max_strength (required range for sigma updates)
        - probability (optional)
        - increase_factor / decrease_factor, diversity thresholds (optional)
        - min_probability / max_probability (optional clamp)

    This class is purely declarative. Strategy-specific calculations and
    state updates occur in the corresponding Para* implementations or update
    helpers.
    """

    strategy: MutationStrategy = Field(..., description="Mutation strategy to use.")

    # Generic / commonly used parameters
    strength: Optional[float] = Field(
        default=None, description="Global mutation strength (sigma)."
    )
    probability: Optional[float] = Field(
        default=None, description="Per-parameter mutation application probability."
    )

    # Exponential / schedule-based starts
    init_strength: Optional[float] = Field(
        default=None, description="Initial strength for schedule-based strategies."
    )
    init_probability: Optional[float] = Field(
        default=None, description="Initial probability for schedule-based strategies."
    )

    # Strength / probability ranges (used for clamping or adaptive updates)
    min_strength: Optional[float] = None
    max_strength: Optional[float] = None

    min_probability: Optional[float] = None
    max_probability: Optional[float] = None

    # Diversity adaptation (optional)
    increase_factor: Optional[float] = Field(
        default=None,
        description="Factor to increase values when diversity is low/high.",
    )
    decrease_factor: Optional[float] = Field(
        default=None,
        description="Factor to decrease values when diversity is low/high.",
    )
    min_diversity_threshold: Optional[float] = None
    max_diversity_threshold: Optional[float] = None

    # Validators

    @model_validator(mode="after")
    def _validate_ranges_and_strategy(self) -> "MutationConfig":
        """Sanity checks for common ranges and strategy-dependent requirements."""
        # Probability bounds (if present)
        for p in (
            self.probability,
            self.init_probability,
            self.min_probability,
            self.max_probability,
        ):
            if p is not None and not (0.0 <= p <= 1.0):
                raise ValueError("All probabilities must be in [0, 1].")

        # Strength non-negativity (if present)
        for s in (
            self.strength,
            self.init_strength,
            self.min_strength,
            self.max_strength,
        ):
            if s is not None and s < 0.0:
                raise ValueError("Mutation strengths must be >= 0.")

        # Min/Max consistency
        if self.min_strength is not None and self.max_strength is not None:
            if self.min_strength > self.max_strength:
                raise ValueError("min_strength must be <= max_strength.")

        if self.min_probability is not None and self.max_probability is not None:
            if self.min_probability > self.max_probability:
                raise ValueError("min_probability must be <= max_probability.")

        # Strategy-specific light requirements
        if self.strategy.name == "CONSTANT":
            if self.strength is None:
                raise ValueError("CONSTANT strategy requires 'strength'.")

        if self.strategy.name == "EXPONENTIAL_DECAY":
            if self.init_strength is None:
                raise ValueError("EXPONENTIAL_DECAY requires 'init_strength'.")

        if self.strategy.name == "ADAPTIVE_GLOBAL":
            if self.init_strength is None or self.init_probability is None:
                raise ValueError(
                    "ADAPTIVE_GLOBAL requires both 'init_strength' and "
                    "'init_probability' as initial values."
                )

        if self.strategy.name in {"ADAPTIVE_INDIVIDUAL", "ADAPTIVE_PER_PARAMETER"}:
            if self.min_strength is None or self.max_strength is None:
                raise ValueError(
                    f"{self.strategy.name} requires 'min_strength' and 'max_strength'."
                )

        return self


class StructuralMutationConfig(BaseModel):
    """Structural mutation configuration (shared across evolvable networks)."""

    add_connection: Optional[float] = 0.0
    remove_connection: Optional[float] = 0.0
    add_neuron: Optional[float] = 0.0
    remove_neuron: Optional[float] = 0.0
    split_connection: Optional[float] = 0.0
    keep_connected: Optional[bool] = True
    max_nodes: Optional[int] = None
    max_edges: Optional[int] = None
    recurrent: Optional[Literal["none", "direct", "local", "all"]] = "none"

    @model_validator(mode="after")
    def _check_ranges(self) -> "StructuralMutationConfig":
        for name in [
            "add_connection",
            "remove_connection",
            "add_neuron",
            "remove_neuron",
            "split_connection",
        ]:
            val = getattr(self, name)
            if val is not None and not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {val}")

        if self.max_nodes is not None and self.max_nodes <= 0:
            raise ValueError("max_nodes must be > 0 or None")

        if self.max_edges is not None and self.max_edges <= 0:
            raise ValueError("max_edges must be > 0 or None")

        if self.recurrent not in {None, "none", "direct", "local", "all"}:
            raise ValueError(f"Invalid value for recurrent: {self.recurrent}")

        return self


class ActivationMutationConfig(BaseModel):
    probability: float = Field(
        ..., description="Per-neuron mutation probability in [0,1]."
    )
    allowed: Optional[list[str]] = Field(
        default=None,
        description="Whitelist of activation names; if None, "
        "all registered activations.",
    )

    @model_validator(mode="after")
    def _check_probability(self) -> "ActivationMutationConfig":
        if not (0.0 <= self.probability <= 1.0):
            raise ValueError("activations.probability must be in [0, 1].")
        return self


class EvoNetMutationConfig(MutationConfig):
    """
    EvoNet-specific variant with optional per-scope overrides.

    Supported override scopes:
        - biases:  MutationConfig for biases  (optional)
        - activations: MutationConfig for activation choices (optional)
        - structural: MutationConfig for structural choices (optional)
    """

    biases: Optional[MutationConfig] = Field(
        default=None, description="Optional override for bias mutation."
    )
    activations: Optional[ActivationMutationConfig] = Field(
        default=None, description="Optional override for activation changes."
    )
    structural: Optional[StructuralMutationConfig] = Field(
        default=None, description="Optional override for structural mutation."
    )

    @model_validator(mode="after")
    def _no_weights_override(self) -> "EvoNetMutationConfig":
        if getattr(self, "weights", None) is not None:
            raise ValueError(
                "mutation.weights is not supported. "
                "Use the global mutation block for weights; "
                "per-scope overrides exist only for biases/activations/structural."
            )
        return self


class CrossoverConfig(BaseModel):
    """
    Configuration block for crossover strategies and operators.

    Strategy (high-level policy) and Operator (low-level mechanism) are modeled
    separately. Depending on the operator, additional parameters may apply.

    Operators (see CrossoverOperator):
        - BLX (uses alpha)
        - SBX (uses eta)
        - INTERMEDIATE (uses blend_range)
        - ...others can be added as needed
    """

    strategy: CrossoverStrategy = Field(..., description="Crossover strategy to use.")
    operator: Optional[CrossoverOperator] = Field(
        default=None, description="Concrete crossover operator (if applicable)."
    )

    # Probability control
    probability: Optional[float] = Field(
        default=None, description="Per-gene/application probability for crossover."
    )
    init_probability: Optional[float] = None
    min_probability: Optional[float] = None
    max_probability: Optional[float] = None

    # Diversity adaptation (optional)
    increase_factor: Optional[float] = None
    decrease_factor: Optional[float] = None

    # Operator-specific parameters
    alpha: Optional[float] = Field(
        default=None, description="BLX-alpha parameter (typ. in [0, 1] or small >1)."
    )
    eta: Optional[float] = Field(
        default=None, description="SBX-eta (non-negative; larger → more local)."
    )
    blend_range: Optional[float] = Field(
        default=None, description="Range for intermediate/blend crossover."
    )

    @model_validator(mode="after")
    def _validate_probability_and_operator(self) -> "CrossoverConfig":
        # Probability bounds (if present)
        for p in (
            self.probability,
            self.init_probability,
            self.min_probability,
            self.max_probability,
        ):
            if p is not None and not (0.0 <= p <= 1.0):
                raise ValueError("All crossover probabilities must be in [0, 1].")

        if self.min_probability is not None and self.max_probability is not None:
            if self.min_probability > self.max_probability:
                raise ValueError("min_probability must be <= max_probability.")

        return self
