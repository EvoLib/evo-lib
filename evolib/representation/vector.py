# SPDX-License-Identifier: MIT

import numpy as np

from evolib.config.vector_component_config import VectorComponentConfig
from evolib.interfaces.enums import (
    CrossoverOperator,
    CrossoverStrategy,
    MutationStrategy,
)
from evolib.interfaces.types import ModuleConfig
from evolib.operators.crossover import (
    crossover_arithmetic,
    crossover_blend_alpha,
    crossover_intermediate,
    crossover_simulated_binary,
)
from evolib.operators.mutation import (
    adapt_mutation_probability_by_diversity,
    adapt_mutation_strength,
    adapt_mutation_strength_by_diversity,
    adapt_mutation_strengths,
    adapted_tau,
    exponential_mutation_probability,
    exponential_mutation_strength,
)
from evolib.representation.base import ParaBase
from evolib.representation.evo_params import EvoControlParams
from evolib.representation.netvector import NetVector


class ParaVector(ParaBase):
    def __init__(self) -> None:

        # Parametervektor
        self.vector: np.ndarray = np.zeros(1)
        self.shape: tuple[int, ...] = (1,)

        self.randomize_mutation_strengths: bool | None = None

        # Bounds of parameter (z. B. [-1, 1])
        self.bounds: tuple[float, float] | None = None
        self.init_bounds: tuple[float, float] | None = None

        # Crossover
        self._crossover_fn = None

        # EvoControlParams
        self.evo_params = EvoControlParams()

    def apply_config(self, cfg: ModuleConfig) -> None:

        if not isinstance(cfg, VectorComponentConfig):
            raise TypeError("Expected VectorComponentConfig")

        evo_params = self.evo_params

        # structure-based interpretation of dimenson
        structure = getattr(cfg, "structure", "flat")

        if structure == "net":
            if not isinstance(cfg.dim, list):
                raise ValueError("structure='net' requires dim as list[int]")
            net = NetVector(dim=cfg.dim, activation=cfg.activation or "tanh")
            cfg.shape = (int(net.n_parameters),)
            cfg.dim = int(net.n_parameters)

        elif structure == "tensor":
            if not isinstance(cfg.dim, list):
                raise ValueError("structure='tensor' requires dim as list[int]")
            cfg.shape = tuple(cfg.dim)
            cfg.dim = int(np.prod(cfg.shape))

        elif structure == "blocks":
            if not isinstance(cfg.dim, list):
                raise ValueError("structure='blocks' requires dim as list[int]")
            cfg.shape = None
            cfg.dim = sum(cfg.dim)

        elif structure == "grouped":
            if not isinstance(cfg.dim, list):
                raise ValueError("structure='grouped' requires dim as list[int]")
            cfg.shape = None
            cfg.dim = sum(cfg.dim)

        elif structure == "flat":
            if isinstance(cfg.dim, list):
                cfg.shape = tuple(cfg.dim)
                cfg.dim = int(np.prod(cfg.shape))
            else:
                cfg.shape = (cfg.dim,)
        else:
            raise ValueError(f"Unknown structure type: '{structure}'")

        # Assign dimensions
        self.dim = cfg.dim
        self.shape = cfg.shape or (cfg.dim,)
        self.vector = np.zeros(self.dim)

        # Bounds
        self.bounds = cfg.bounds
        self.init_bounds = cfg.init_bounds or self.bounds

        # Mutation
        mutation_cfg = cfg.mutation
        if mutation_cfg is None:
            raise ValueError("Mutation config is required for ParaVector.")
        evo_params.tau = cfg.tau or 0.0
        self.randomize_mutation_strengths = cfg.randomize_mutation_strengths or False

        evo_params.mutation_strategy = mutation_cfg.strategy

        # Strategy-specific mutation params
        if evo_params.mutation_strategy == MutationStrategy.CONSTANT:
            evo_params.mutation_probability = mutation_cfg.probability
            evo_params.mutation_strength = mutation_cfg.strength

        elif evo_params.mutation_strategy == MutationStrategy.EXPONENTIAL_DECAY:
            evo_params.min_mutation_probability = mutation_cfg.min_probability
            evo_params.max_mutation_probability = mutation_cfg.max_probability
            evo_params.min_mutation_strength = mutation_cfg.min_strength
            evo_params.max_mutation_strength = mutation_cfg.max_strength

        elif evo_params.mutation_strategy == MutationStrategy.ADAPTIVE_GLOBAL:
            evo_params.mutation_probability = mutation_cfg.init_probability
            evo_params.mutation_strength = mutation_cfg.init_strength
            evo_params.min_mutation_probability = mutation_cfg.min_probability
            evo_params.max_mutation_probability = mutation_cfg.max_probability
            evo_params.min_mutation_strength = mutation_cfg.min_strength
            evo_params.max_mutation_strength = mutation_cfg.max_strength
            evo_params.min_diversity_threshold = mutation_cfg.min_diversity_threshold
            evo_params.max_diversity_threshold = mutation_cfg.max_diversity_threshold
            evo_params.mutation_inc_factor = mutation_cfg.increase_factor
            evo_params.mutation_dec_factor = mutation_cfg.decrease_factor

        elif evo_params.mutation_strategy == MutationStrategy.ADAPTIVE_INDIVIDUAL:
            evo_params.min_mutation_strength = mutation_cfg.min_strength
            evo_params.max_mutation_strength = mutation_cfg.max_strength

        elif evo_params.mutation_strategy == MutationStrategy.ADAPTIVE_PER_PARAMETER:
            evo_params.min_mutation_strength = mutation_cfg.min_strength
            evo_params.max_mutation_strength = mutation_cfg.max_strength

        else:
            raise ValueError(
                f"Unknown mutation strategy: {evo_params.mutation_strategy}"
            )

        # Crossover
        crossover_cfg = cfg.crossover
        if crossover_cfg is None:
            evo_params.crossover_strategy = CrossoverStrategy.NONE
        else:
            evo_params.crossover_strategy = crossover_cfg.strategy
            evo_params.crossover_probability = (
                crossover_cfg.probability or crossover_cfg.init_probability
            )

            evo_params.min_crossover_probability = crossover_cfg.min_probability
            evo_params.max_crossover_probability = crossover_cfg.max_probability
            evo_params.crossover_inc_factor = crossover_cfg.increase_factor
            evo_params.crossover_dec_factor = crossover_cfg.decrease_factor

            op = crossover_cfg.operator
            if op == CrossoverOperator.BLX:
                alpha = crossover_cfg.alpha or 0.5
                self._crossover_fn = lambda a, b: crossover_blend_alpha(a, b, alpha)
            elif op == CrossoverOperator.ARITHMETIC:
                self._crossover_fn = crossover_arithmetic
            elif op == CrossoverOperator.SBX:
                eta = crossover_cfg.eta or 15.0
                self._crossover_fn = lambda a, b: crossover_simulated_binary(a, b, eta)
            elif op == CrossoverOperator.INTERMEDIATE:
                blend = crossover_cfg.blend_range or 0.25
                self._crossover_fn = lambda a, b: crossover_intermediate(a, b, blend)
            else:
                self._crossover_fn = None

    def mutate(self) -> None:
        """
        Applies Gaussian mutation to the parameter vector.

        If `mutation_strengths` is defined, per-parameter mutation is used.
        Otherwise, global mutation strength and optional mutation probability
        determine mutation behavior.
        """

        if self.evo_params.mutation_strengths is not None:

            # Adaptive per-parameter mutation
            noise = np.random.normal(
                loc=0.0, scale=self.evo_params.mutation_strengths, size=len(self.vector)
            )

            self.vector += noise
        else:
            if self.evo_params.mutation_strength is None:
                raise ValueError("mutation_strength must be set.")
            # Global mutation path (scalar mutation_strength required)
            noise = np.random.normal(
                loc=0.0, scale=self.evo_params.mutation_strength, size=self.vector.shape
            )
            prob = self.evo_params.mutation_probability or 1.0
            mask = (np.random.rand(len(self.vector)) < prob).astype(np.float64)
            self.vector += noise * mask

        if self.bounds is not None:
            self.vector = np.clip(self.vector, *self.bounds)

    def print_status(self) -> None:
        status = self.get_status()
        print(status)

    def get_status(self) -> str:
        """Returns a formatted string summarizing the internal state of the
        ParaVector."""
        parts = []

        vector_preview = np.round(self.vector[:4], 3).tolist()
        parts.append(f"Vector={vector_preview}{'...' if len(self.vector) > 4 else ''}")

        if self.evo_params.mutation_strength is not None:
            parts.append(
                f"Global mutation_strength=" f"{self.evo_params.mutation_strength:.4f}"
            )

        if self.evo_params.crossover_probability is not None:
            parts.append(f"crossover_prob={self.evo_params.crossover_probability:.4f}")

        if self.evo_params.tau != 0.0:
            parts.append(f"tau={self.evo_params.tau:.4f}")

        if self.evo_params.mutation_strengths is not None:
            parts.append(
                f"Para mutation strength: "
                f"mean={np.mean(self.evo_params.mutation_strengths):.4f}, "
                f"min={np.min(self.evo_params.mutation_strengths):.4f}, "
                f"max={np.max(self.evo_params.mutation_strengths):.4f}"
            )

        return " | ".join(parts)

    def get_history(self) -> dict[str, float]:
        """
        Return a dictionary of internal mutation-relevant values for logging.

        This supports both global and per-parameter adaptive strategies.
        """
        history = {}

        # global updatefaktor
        if self.evo_params.tau is not None:
            history["tau"] = float(self.evo_params.tau)

        # globale mutationstregth (optional)
        if self.evo_params.mutation_strength is not None:
            history["mutation_strength"] = float(self.evo_params.mutation_strength)

        # vector mutationsstrength
        if self.evo_params.mutation_strengths is not None:
            strengths = self.evo_params.mutation_strengths
            history.update(
                {
                    "sigma_mean": float(np.mean(strengths)),
                    "sigma_min": float(np.min(strengths)),
                    "sigma_max": float(np.max(strengths)),
                }
            )

        return history

    def update_mutation_parameters(
        self, generation: int, max_generations: int, diversity_ema: float | None = None
    ) -> None:

        ep = self.evo_params
        """Update mutation parameters based on strategy and generation."""
        if ep.mutation_strategy == MutationStrategy.EXPONENTIAL_DECAY:
            ep.mutation_strength = exponential_mutation_strength(
                ep, generation, max_generations
            )

            ep.mutation_probability = exponential_mutation_probability(
                ep, generation, max_generations
            )

        elif ep.mutation_strategy == MutationStrategy.ADAPTIVE_GLOBAL:
            if diversity_ema is None:
                raise ValueError(
                    "diversity_ema must be provided" "for ADAPTIVE_GLOBAL strategy"
                )
            if ep.mutation_strength is None:
                raise ValueError(
                    "mutation_strength must be provided" "for ADAPTIVE_GLOBAL strategy"
                )
            if ep.mutation_probability is None:
                raise ValueError(
                    "mutation_probability must be provided"
                    "for ADAPTIVE_GLOBAL strategy"
                )

            ep.mutation_probability = adapt_mutation_probability_by_diversity(
                ep.mutation_probability, diversity_ema, ep
            )

            ep.mutation_strength = adapt_mutation_strength_by_diversity(
                ep.mutation_strength, diversity_ema, ep
            )

        elif ep.mutation_strategy == MutationStrategy.ADAPTIVE_INDIVIDUAL:
            # Ensure tau is initialized
            ep.tau = adapted_tau(len(self.vector))

            if ep.min_mutation_strength is None or ep.max_mutation_strength is None:
                raise ValueError(
                    "min_mutation_strength and max_mutation_strength" "must be defined."
                )
            if self.bounds is None:
                raise ValueError("bounds must be set")
            # Ensure mutation_strength is initialized
            if ep.mutation_strength is None:
                ep.mutation_strength = np.random.uniform(
                    ep.min_mutation_strength, ep.max_mutation_strength
                )

            # Perform adaptive update
            ep.mutation_strength = adapt_mutation_strength(ep, self.bounds)

        elif ep.mutation_strategy == MutationStrategy.ADAPTIVE_PER_PARAMETER:
            if ep.tau == 0.0 or ep.tau is None:
                ep.tau = adapted_tau(len(self.vector))

            if ep.min_mutation_strength is None or ep.max_mutation_strength is None:
                raise ValueError(
                    "min_mutation_strength and max_mutation_strength" "must be defined."
                )

            if self.bounds is None:
                raise ValueError("bounds must be set")

            if ep.mutation_strengths is None:
                ep.mutation_strengths = np.random.uniform(
                    ep.min_mutation_strength,
                    ep.max_mutation_strength,
                    size=len(self.vector),
                )

            # Perform adaptive update
            ep.mutation_strengths = adapt_mutation_strengths(ep, self.bounds)

    def crossover_with(self, partner: "ParaBase") -> None:
        """
        Applies crossover with another ParaBase-compatible instance.

        This method is specific to ParaVector and expects the partner to also be a
        ParaVector. The internal _crossover_fn may return either a single offspring
        vector or a tuple of two.
        """
        if not isinstance(partner, ParaVector):
            return

        if self._crossover_fn is None:
            return

        result = self._crossover_fn(self.vector, partner.vector)

        if isinstance(result, tuple):
            child1, child2 = result
        else:
            child1 = child2 = result

        if self.bounds is None or partner.bounds is None:
            raise ValueError("Both participants must define bounds before crossover.")

        min_val, max_val = self.bounds
        self.vector = np.clip(child1, min_val, max_val)

        min_val_p, max_val_p = partner.bounds
        partner.vector = np.clip(child2, min_val_p, max_val_p)
