# SPDX-License-Identifier: MIT
"""
Shared configuration blocks for mutation and crossover used across multiple
ComponentConfig classes (e.g. VectorComponentConfig, EvoNetComponentConfig).

The classes here are intentionally small and reusable: they describe *what*
should be configured, not *how* it is executed. Any runtime behavior belongs
into the respective Para* representations and operator modules.
"""

from typing import Literal, Optional, Union

from evonet.activation import ACTIVATIONS
from pydantic import BaseModel, ConfigDict, Field, model_validator, validator

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

    model_config = ConfigDict(extra="forbid")

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
                raise ValueError("CONSTANT requires 'strength'.")
            if self.init_strength is not None or self.init_probability is not None:
                raise ValueError("CONSTANT must not define init_* fields.")

        if self.strategy.name == "EXPONENTIAL_DECAY":
            if self.init_strength is None:
                raise ValueError("EXPONENTIAL_DECAY requires 'init_strength'.")
            if self.strength is not None:
                raise ValueError("EXPONENTIAL_DECAY must not define 'strength'.")

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


class AddNeuron(BaseModel):
    probability: float = Field(0.0, ge=0.0, le=1.0)
    init_connection_ratio: Optional[float] = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Fraction of allowed connections to actually add [0.0, 1.0]).",
    )
    init: Optional[Literal["none", "zero", "near_zero", "random"]] = "zero"

    # Whitelist for random activation selection
    activations_allowed: Optional[list[str]] = Field(
        default=None,
        description="Whitelist of activation names applied to "
        "neurons in hidden layers.",
    )

    @validator("activations_allowed", each_item=True)
    def validate_activation_name(cls, act_name: str) -> str:
        """Ensure only valid activation function names are allowed."""
        if act_name not in ACTIVATIONS:
            raise ValueError(
                f"Invalid activation function '{act_name}'. "
                f"Valid options are: {list(ACTIVATIONS.keys())}"
            )
        return act_name


class RemoveNeuron(BaseModel):
    probability: float = Field(0.0, ge=0.0, le=1.0)


class AddConnection(BaseModel):
    probability: float = Field(0.0, ge=0.0, le=1.0)
    max: int = Field(
        default=1,
        ge=1,
        description=(
            "Maximum number of new connections to add per structural mutation event."
        ),
    )
    init: Optional[Literal["none", "zero", "near_zero", "random"]] = "zero"


class RemoveConnection(BaseModel):
    probability: float = Field(0.0, ge=0.0, le=1.0)
    max: int = Field(
        default=1,
        ge=1,
        description=(
            "Maximum number of connections to remove per structural mutation event."
        ),
    )


class StructuralTopology(BaseModel):
    """
    Configuration of global structural constraints for EvoNet.

    These parameters regulate the *allowed topology* of the network, independent of
    mutation operators.
    """

    recurrent: Optional[Literal["none", "direct", "local", "all"]] = "none"
    connection_scope: Optional[Literal["adjacent", "crosslayer"]] = "adjacent"

    max_neurons: Optional[int] = Field(
        default=None,
        description="Upper bound on the number of neurons allowed in the network.",
        ge=1,
    )
    max_connections: Optional[int] = Field(
        default=None,
        description="Upper bound on total number of connections "
        "allowed in the network.",
        ge=1,
    )

    @model_validator(mode="after")
    def _validate_topology(self) -> "StructuralTopology":
        if self.recurrent not in {None, "none", "direct", "local", "all"}:
            raise ValueError(f"Invalid recurrent value: {self.recurrent}")

        if self.connection_scope not in {None, "adjacent", "crosslayer"}:
            raise ValueError(f"Invalid connection_scope: {self.connection_scope}")

        # Only positive limits allowed
        if self.max_connections is not None and self.max_connections <= 0:
            raise ValueError("max_connections must be > 0")

        if self.max_neurons is not None and self.max_neurons <= 0:
            raise ValueError("max_neurons must be > 0")

        return self


class StructuralMutationConfig(BaseModel):
    """Structural mutation configuration (shared across evolvable networks)."""

    add_neuron: Optional[AddNeuron] = Field(
        default=None, description="Optional for structural mutation."
    )
    remove_neuron: Optional[RemoveNeuron] = Field(
        default=None, description="Optional for structural mutation."
    )
    add_connection: Optional[AddConnection] = Field(
        default=None, description="Optional for structural mutation."
    )
    remove_connection: Optional[RemoveConnection] = Field(
        default=None, description="Optional for structural mutation."
    )
    topology: StructuralTopology = Field(
        default_factory=StructuralTopology,
        description="Global structural constraints independent of mutation operators.",
    )

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _check_ranges(self) -> "StructuralMutationConfig":
        # Block-based mutation probability
        if self.add_neuron is not None:
            probability = self.add_neuron.probability
            if not (0.0 <= probability <= 1.0):
                raise ValueError(
                    f"add_neuron.probability must be in [0, 1], got {probability}"
                )
            # Initial Connection Ratio
            if self.add_neuron.init_connection_ratio is not None:
                ratio = self.add_neuron.init_connection_ratio
                if not (0.0 <= ratio <= 1.0):
                    raise ValueError(
                        f"init_connection_ratio must be " f"in [0, 1], got {ratio}"
                    )

        if self.remove_neuron is not None:
            probability = self.remove_neuron.probability
            if not (0.0 <= probability <= 1.0):
                raise ValueError(
                    f"remove_neuron.probability must be in [0, 1], got {probability}"
                )
        if self.add_connection is not None:
            probability = self.add_connection.probability
            if not (0.0 <= probability <= 1.0):
                raise ValueError(
                    f"add_connection.probability must be in [0, 1], got {probability}"
                )
            max = self.add_connection.max
            if max < 1:
                raise ValueError(f"add_connection.max must be >= 1, got {max}")

        if self.remove_connection is not None:
            probability = self.remove_connection.probability
            if not (0.0 <= probability <= 1.0):
                raise ValueError(
                    f"remove_connection.probability must be in [0, 1], "
                    f"got {probability}"
                )
            max = self.remove_connection.max
            if max < 1:
                raise ValueError(f"remove_connection.max must be >= 1, got {max}")

        return self


class ActivationMutationConfig(BaseModel):

    model_config = ConfigDict(extra="forbid")

    probability: float = Field(
        ..., description="Per-neuron mutation probability in [0,1]."
    )
    allowed: Optional[list[str]] = Field(
        default=None,
        description="Global whitelist of activation names; "
        "applies to all hidden layers.",
    )
    layers: Optional[dict[int, Union[list[str], Literal["all"]]]] = Field(
        default=None,
        description="Optional mapping from layer index to allowed activations. "
        "Overrides `allowed` if given.",
    )

    @model_validator(mode="after")
    def _check_valid(self) -> "ActivationMutationConfig":
        if not (0.0 <= self.probability <= 1.0):
            raise ValueError("activations.probability must be in [0, 1].")

        if self.allowed is not None and self.layers is not None:
            # Warn about ambiguity, or raise (designentscheidung)
            raise ValueError("Specify either `allowed` or `layers`, not both.")

        return self


class EvoNetMutationConfig(MutationConfig):
    """
    EvoNet-specific variant with optional per-scope overrides.

    Supported override scopes:
        - biases:  MutationConfig for biases  (optional)
        - activations: MutationConfig for activation choices (optional)
        - structural: MutationConfig for structural choices (optional)
    """

    model_config = ConfigDict(extra="forbid")

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

    model_config = ConfigDict(extra="forbid")

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
        default=None, description="SBX-eta (non-negative; larger -> more local)."
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


class HeliConfig(BaseModel):
    """
    HELI (Hierarchical Evolution with Lineage Incubation) configuration block.

    Defines how local micro-evolutions for structure-mutated individuals are run.

    Parameters:

    generations : int
        Number of micro-evolution generations per seed individual.
    offspring_per_seed : int
        Number of offspring to create per incubated seed (λᵢ).
    max_fraction : float
        Upper fraction of offspring eligible for incubation (0–1).
    reduce_sigma_factor : float
        Damping factor for mutation strength during incubation
        (<1.0 -> less exploration).
    drift_stop_above : float, optional
        Abort incubation if drift exceeds this value (seed too poor).
    drift_stop_below : float, optional
        Abort incubation if drift goes below this value (seed already good).
    """

    model_config = ConfigDict(extra="forbid")

    generations: int = Field(
        5,
        ge=1,
        description="Number of local generations each incubation subpopulation runs.",
    )
    offspring_per_seed: int = Field(
        10,
        ge=1,
        description="Number of offspring generated per incubated seed individual.",
    )
    max_fraction: float = Field(
        0.1,
        ge=0.0,
        le=1.0,
        description="Maximum fraction of offspring that can enter incubation (0–1).",
    )
    reduce_sigma_factor: float = Field(
        0.5,
        ge=0.0,
        description="Damping factor for mutation strength during incubation"
        "(<1.0 reduces it).",
    )

    drift_stop_above: Optional[float] = Field(
        default=None,
        description="Abort incubation if drift > threshold (too poor).",
    )
    drift_stop_below: Optional[float] = Field(
        default=None,
        description="Abort incubation if drift < threshold (already viable).",
    )

    @model_validator(mode="after")
    def _check_consistency(self) -> "HeliConfig":
        """Ensure all values are within valid ranges and logically consistent."""
        if self.generations <= 0:
            raise ValueError("generations must be > 0")
        if self.offspring_per_seed <= 0:
            raise ValueError("offspring_per_seed must be > 0")
        if not (0.0 <= self.max_fraction <= 1.0):
            raise ValueError("max_fraction must be between 0 and 1.")
        if self.reduce_sigma_factor < 0.0:
            raise ValueError("reduce_sigma_factor must be >= 0.")
        return self
