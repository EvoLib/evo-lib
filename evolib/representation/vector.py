import numpy as np

from evolib.interfaces.enums import CrossoverStrategy, MutationStrategy
from evolib.interfaces.structs import MutationParams
from evolib.operators.mutation import adapted_mutation_strength
from evolib.representation.base import ParaBase


class ParaVector(ParaBase):
    def __init__(self) -> None:

        # Mutationstrategy
        self.mutation_strategy: MutationStrategy | None = None

        # Global Mutationparameter
        self.mutation_strength: float | None = None
        self.mutation_probability: float | None = None
        self.tau: float = 0.0

        # Per-Parameter Mutationparameter
        self.para_mutation_strengths: np.ndarray | None = None
        self.randomize_mutation_strengths: bool | None = None

        # Bounds of parameter (z. B. [-1, 1])
        self.bounds: tuple[float, float] | None = None
        self.init_bounds: tuple[float, float] | None = None

        # Parametervektor
        self.vector: np.ndarray = np.zeros(1)

        # Bounds for mutation (min/max)
        self.min_mutation_strength: float | None = None
        self.max_mutation_strength: float | None = None
        self.min_mutation_probability: float | None = None
        self.max_mutation_probability: float | None = None

        # Diversity based Adaptionfaktors
        self.mutation_inc_factor: float | None = None
        self.mutation_dec_factor: float | None = None
        self.min_diversity_threshold: float | None = None
        self.max_diversity_threshold: float | None = None

        # Crossover
        self.crossover_strategy: CrossoverStrategy | None = None
        self.crossover_probability: float | None = None
        self.min_crossover_probability: float | None = None
        self.max_crossover_probability: float | None = None
        self.crossover_inc_factor: float | None = None
        self.crossover_dec_factor: float | None = None

    def mutate(self) -> None:
        """
        Applies Gaussian mutation to the parameter vector.

        If `para_mutation_strengths` is defined, per-parameter mutation is used.
        Otherwise, global mutation strength and optional mutation probability
        determine mutation behavior.

        Ensures type safety and runtime correctness.
        """

        if self.para_mutation_strengths is not None:
            # Adaptive per-parameter mutation
            noise = np.random.normal(
                loc=0.0,
                scale=self.para_mutation_strengths,
                size=len(self.vector),
            )
            self.vector += noise

        else:
            # Global mutation path (scalar mutation_strength required)
            if self.mutation_strength is None:
                raise ValueError(
                    "mutation_strength must be defined for global mutation."
                )

            noise = np.random.normal(
                loc=0.0,
                scale=self.mutation_strength,
                size=len(self.vector),
            )

            prob = self.mutation_probability or 1.0  # garantiert float
            mask = (np.random.rand(len(self.vector)) < prob).astype(np.float64)
            self.vector += noise * mask

        # Clip to bounds
        assert self.bounds is not None
        self.vector = np.clip(self.vector, *self.bounds)

    def update_tau(self) -> None:
        """
        Update the learning rate tau based on the vector length.

        This implements a simple self-adaptation rule:
        tau = 1 / sqrt(n), where n = number of parameters.
        """
        if self.tau is not None:
            n = len(self.vector)
            self.tau = 1.0 / np.sqrt(n) if n > 0 else 0.0

    def adapt_mutation_strength(self, params: MutationParams) -> None:
        """
        Applies a log-normal update to the global mutation strength.

        This method is only applicable if `mutation_strength` is defined as a scalar
        attribute of this ParaVector instance.

        Args:
            params (MutationParams): Contains bounds for clipping the updated strength,
                in particular `min_strength` and `max_strength`.

        Raises:
            AttributeError: If `mutation_strength` is not defined in this instance.
        """
        if not hasattr(self, "mutation_strength"):
            raise AttributeError("mutation_strength not defined in this ParaVector.")

        self.mutation_strength = adapted_mutation_strength(params)

    def adapt_para_mutation_strengths(self, params: MutationParams) -> None:
        """
        Adapt each gene-specific mutation strength using independent log-normal factors.

        This corresponds to the self-adaptive mutation strategy where each parameter
        dimension maintains its own mutation strength (sigma_i).
        """

        if self.para_mutation_strengths is None:
            raise ValueError(
                "para_mutation_strengths must be initialized" "before adaptation."
            )

        self.para_mutation_strengths *= np.exp(
            self.tau * np.random.normal(size=len(self.vector))
        )
        self.para_mutation_strengths = np.clip(
            self.para_mutation_strengths, params.min_strength, params.max_strength
        )

    def print_status(self) -> None:
        status = self.get_status()
        print(status)

    def get_status(self) -> str:
        """Returns a formatted string summarizing the internal state of the
        ParaVector."""
        parts = []

        vector_preview = np.round(self.vector[:4], 3).tolist()
        parts.append(f"Vector={vector_preview}{'...' if len(self.vector) > 4 else ''}")

        if hasattr(self, "mutation_strength") and self.mutation_strength is not None:
            parts.append(f"Global mutation_strength={self.mutation_strength:.4f}")

        if hasattr(self, "tau") and self.tau != 0.0:
            parts.append(f"tau={self.tau:.4f}")

        if self.para_mutation_strengths is not None:
            para_mutation_strengths = self.para_mutation_strengths
            parts.append(
                f"Para mutation strength: mean={np.mean(para_mutation_strengths):.4f}, "
                f"min={np.min(para_mutation_strengths):.4f}, "
                f"max={np.max(para_mutation_strengths):.4f}"
            )

        return " | ".join(parts)

    def get_history(self) -> dict[str, float]:
        """
        Return a dictionary of internal mutation-relevant values for logging.

        This supports both global and per-parameter adaptive strategies.
        """
        history = {}

        # global updatefaktor
        if hasattr(self, "tau"):
            history["tau"] = float(self.tau)

        # globale mutationstregth (optional)
        if hasattr(self, "mutation_strength") and self.mutation_strength is not None:
            history["mutation_strength"] = float(self.mutation_strength)

        # vector mutationsstrength
        if self.para_mutation_strengths is not None:
            strengths = self.para_mutation_strengths
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
        """Update mutation parameters based on strategy and generation."""
        if self.mutation_strategy == MutationStrategy.EXPONENTIAL_DECAY:
            self.mutation_strength = self._exponential_mutation_strength(
                generation, max_generations
            )
            self.mutation_probability = self._exponential_mutation_probability(
                generation, max_generations
            )

        elif self.mutation_strategy == MutationStrategy.ADAPTIVE_GLOBAL:
            if diversity_ema is None:
                raise ValueError(
                    "diversity_ema must be provided" "for ADAPTIVE_GLOBAL strategy"
                )
            self.mutation_probability = self._adaptive_mutation_probability(
                diversity_ema
            )
            self.mutation_strength = self._adaptive_mutation_strength(diversity_ema)

        elif self.mutation_strategy == MutationStrategy.ADAPTIVE_INDIVIDUAL:
            # Ensure tau is initialized
            self.update_tau()

            if self.min_mutation_strength is None or self.max_mutation_strength is None:
                raise ValueError(
                    "min_mutation_strength and max_mutation_strength" "must be defined."
                )
            if self.bounds is None:
                raise ValueError("bounds must be set")
            # Ensure mutation_strength is initialized
            if self.mutation_strength is None:
                self.mutation_strength = np.random.uniform(
                    self.min_mutation_strength, self.max_mutation_strength
                )

            # Perform adaptive update
            params = MutationParams(
                strength=self.mutation_strength,
                min_strength=self.min_mutation_strength,
                max_strength=self.max_mutation_strength,
                probability=1.0,  # unused here
                min_probability=0.0,  # unused here
                max_probability=0.0,  # unused here
                bounds=self.bounds,
                bias=None,
                tau=self.tau,
            )

            self.adapt_mutation_strength(params)

        elif self.mutation_strategy == MutationStrategy.ADAPTIVE_PER_PARAMETER:
            if self.tau == 0.0 or self.tau is None:
                self.update_tau()

            if self.min_mutation_strength is None or self.max_mutation_strength is None:
                raise ValueError(
                    "min_mutation_strength and max_mutation_strength" "must be defined."
                )

            if self.bounds is None:
                raise ValueError("bounds must be set")

            if self.para_mutation_strengths is None:
                self.para_mutation_strengths = np.random.uniform(
                    self.min_mutation_strength,
                    self.max_mutation_strength,
                    size=len(self.vector),
                )

            params = MutationParams(
                strength=1.0,  # unused
                min_strength=self.min_mutation_strength,
                max_strength=self.max_mutation_strength,
                probability=1.0,  # unused
                min_probability=0.0,  # unused here
                max_probability=0.0,  # unused here
                bounds=self.bounds,
                tau=self.tau,
            )

            self.adapt_para_mutation_strengths(params)

    def update_crossover_parameters(
        self, generation: int, max_generations: int, diversity_ema: float | None = None
    ) -> None:
        """
        Update crossover parameters based on strategy and diversity (if applicable).

        Supports exponential decay and adaptive global crossover strategies.
        """
        if self.crossover_strategy == CrossoverStrategy.EXPONENTIAL_DECAY:
            self.crossover_probability = self._exponential_crossover_probability(
                generation, max_generations
            )

        elif self.crossover_strategy == CrossoverStrategy.ADAPTIVE_GLOBAL:
            if diversity_ema is None:
                raise ValueError(
                    "diversity_ema must be provided for ADAPTIVE_GLOBAL"
                    "crossover strategy"
                )

            self.crossover_probability = self._adaptive_crossover_probability(
                diversity_ema
            )

    def _exponential_mutation_strength(
        self, generation: int, max_generations: int
    ) -> float:
        """
        Calculates exponentially decaying mutation strength over generations.

        Args:
            generation: int:

        Returns:
            float: The adjusted mutation strength.
        """
        if self.min_mutation_strength is None or self.max_mutation_strength is None:
            raise ValueError(
                "min_mutation_strength and max_mutation_strength" "must be defined."
            )
        k = (
            np.log(self.max_mutation_strength / self.min_mutation_strength)
            / max_generations
        )
        return self.max_mutation_strength * np.exp(-k * generation)

    def _exponential_mutation_probability(
        self, generation: int, max_generations: int
    ) -> float:
        """
        Calculates exponentially decaying mutation probablility over generations.

        Args:
            generation: int

        Returns:
            float: The adjusted mutation rate.
        """
        if (
            self.min_mutation_probability is None
            or self.max_mutation_probability is None
        ):
            raise ValueError(
                "min_mutation_probability and max_mutation_probability"
                "must be defined."
            )
        k = (
            np.log(self.max_mutation_probability / self.min_mutation_probability)
            / max_generations
        )
        return self.max_mutation_probability * np.exp(-k * generation)

    def _adaptive_mutation_strength(self, diversity_ema: float) -> float:
        """
        Calculates adapted mutation strength based on population diversity EMA.

        Uses configured thresholds and scaling factors.

        Args:
            diversity_ema (float): Exponentially smoothed population diversity.

        Returns:
            float: Updated mutation strength.
        """
        if self.min_diversity_threshold is None or self.max_diversity_threshold is None:
            raise ValueError(
                "min_diversity_threshold and min_diversity_threshold" "must be defined."
            )
        if self.min_mutation_strength is None or self.max_mutation_strength is None:
            raise ValueError(
                "min_mutation_strength and max_mutation_strength" "must be defined."
            )
        if self.mutation_strength is None:
            raise ValueError("mutation_strength must be defined for global mutation.")
        if self.mutation_inc_factor is None or self.mutation_dec_factor is None:
            raise ValueError(
                "mutation_inc_factor and mutation_dec_factor" "must be defined."
            )

        if diversity_ema < self.min_diversity_threshold:
            return min(
                self.max_mutation_strength,
                self.mutation_strength * self.mutation_inc_factor,
            )
        elif diversity_ema > self.max_diversity_threshold:
            return max(
                self.min_mutation_strength,
                self.mutation_strength * self.mutation_dec_factor,
            )
        else:
            return self.mutation_strength

    def _adaptive_mutation_probability(self, diversity_ema: float) -> float:
        """
        Calculates adapted mutation probability based on population diversity EMA.

        Uses configured thresholds and scaling factors.

        Args:
            diversity_ema (float): Exponentially smoothed population diversity.

        Returns:
            float: Updated mutation probability.
        """
        if self.min_diversity_threshold is None or self.max_diversity_threshold is None:
            raise ValueError(
                "min_diversity_threshold and min_diversity_threshold" "must be defined."
            )
        if (
            self.min_mutation_probability is None
            or self.max_mutation_probability is None
        ):
            raise ValueError(
                "min_mutation_probability and max_mutation_probability"
                "must be defined."
            )
        if self.mutation_probability is None:
            raise ValueError(
                "mutation_probability must be defined for global mutation."
            )
        if self.mutation_inc_factor is None or self.mutation_dec_factor is None:
            raise ValueError(
                "mutation_inc_factor and mutation_dec_factor" "must be defined."
            )

        if diversity_ema < self.min_diversity_threshold:
            return min(
                self.max_mutation_probability,
                self.mutation_probability * self.mutation_inc_factor,
            )
        elif diversity_ema > self.max_diversity_threshold:
            return max(
                self.min_mutation_probability,
                self.mutation_probability * self.mutation_dec_factor,
            )
        else:
            return self.mutation_probability

    def _adaptive_crossover_probability(self, diversity_ema: float) -> float:
        """
        Calculates adapted crossover probability based on population diversity EMA.

        Uses configured thresholds and scaling factors.

        Args:
            diversity_ema (float): Exponentially smoothed population diversity.

        Returns:
            float: Updated crossover probability.
        """
        if self.min_diversity_threshold is None or self.max_diversity_threshold is None:
            raise ValueError(
                "min_diversity_threshold and max_diversity_threshold" "must be defined."
            )

        if (
            self.min_crossover_probability is None
            or self.max_crossover_probability is None
        ):
            raise ValueError(
                "min_crossover_probability and max_crossover_probability"
                "must be defined."
            )

        if self.crossover_probability is None:
            raise ValueError("crossover_probability must be defined.")

        if self.crossover_inc_factor is None or self.crossover_dec_factor is None:
            raise ValueError(
                "crossover_inc_factor and crossover_dec_factor must be" "defined."
            )

        if diversity_ema < self.min_diversity_threshold:
            return min(
                self.max_crossover_probability,
                self.crossover_probability * self.crossover_inc_factor,
            )
        elif diversity_ema > self.max_diversity_threshold:
            return max(
                self.min_crossover_probability,
                self.crossover_probability * self.crossover_dec_factor,
            )
        else:
            return self.crossover_probability

    def _exponential_crossover_probability(
        self, generation: int, max_generations: int
    ) -> float:

        if (
            self.min_crossover_probability is None
            or self.max_crossover_probability is None
        ):
            raise ValueError(
                "min_crossover_probability and max_crossover_probability"
                "must be defined."
            )
        k = (
            np.log(self.max_crossover_probability / self.min_crossover_probability)
            / max_generations
        )
        return self.max_crossover_probability * np.exp(-k * generation)

    def apply_config(self, cfg: dict) -> None:
        """Apply configuration dictionary to this ParaVector instance."""
        representation_cfg = cfg.get("representation", {})
        self.representation = representation_cfg["type"]
        self.dim = representation_cfg["dim"]
        self.tau = representation_cfg.get("tau", 0.0)
        self.bounds = representation_cfg["bounds"]
        self.init_bounds = representation_cfg.get("init_bounds", self.bounds)

        # Mutation
        mutation_cfg = cfg.get("mutation", {})
        self.mutation_strategy = MutationStrategy(
            mutation_cfg.get("strategy", "constant")
        )

        if self.mutation_strategy == MutationStrategy.CONSTANT:
            self.mutation_probability = mutation_cfg["probability"]
            self.mutation_strength = mutation_cfg["strength"]

        elif self.mutation_strategy == MutationStrategy.EXPONENTIAL_DECAY:
            self.min_mutation_probability = mutation_cfg["min_probability"]
            self.max_mutation_probability = mutation_cfg["max_probability"]
            self.min_mutation_strength = mutation_cfg["min_strength"]
            self.max_mutation_strength = mutation_cfg["max_strength"]

        elif self.mutation_strategy == MutationStrategy.ADAPTIVE_GLOBAL:
            self.mutation_probability = mutation_cfg["init_probability"]
            self.min_mutation_probability = mutation_cfg["min_probability"]
            self.max_mutation_probability = mutation_cfg["max_probability"]

            self.mutation_strength = mutation_cfg["init_strength"]
            self.min_mutation_strength = mutation_cfg["min_strength"]
            self.max_mutation_strength = mutation_cfg["max_strength"]

            self.min_diversity_threshold = mutation_cfg["min_diversity_threshold"]
            self.max_diversity_threshold = mutation_cfg["max_diversity_threshold"]

            self.mutation_inc_factor = mutation_cfg["increase_factor"]
            self.mutation_dec_factor = mutation_cfg["decrease_factor"]

        elif self.mutation_strategy == MutationStrategy.ADAPTIVE_INDIVIDUAL:
            self.mutation_probability = None
            self.mutation_strength = None

            self.min_mutation_strength = mutation_cfg["min_strength"]
            self.max_mutation_strength = mutation_cfg["max_strength"]
            self.update_tau()

        elif self.mutation_strategy == MutationStrategy.ADAPTIVE_PER_PARAMETER:
            self.mutation_probability = None
            self.mutation_strength = None
            self.min_mutation_probability = None
            self.max_mutation_probability = None
            self.min_mutation_strength = mutation_cfg["min_strength"]
            self.max_mutation_strength = mutation_cfg["max_strength"]
            self.randomize_mutation_strengths = representation_cfg.get(
                "randomize_mutation_strengths", False
            )

        # Crossover
        crossover_cfg = cfg.get("crossover", None)
        if crossover_cfg is None:
            self.crossover_strategy = CrossoverStrategy.NONE
            self.crossover_probability = None
        else:
            self.crossover_strategy = CrossoverStrategy(cfg["crossover"]["strategy"])
            if self.crossover_strategy == CrossoverStrategy.CONSTANT:
                self.crossover_probability = cfg["crossover"]["probability"]

            elif self.crossover_strategy == CrossoverStrategy.EXPONENTIAL_DECAY:
                self.min_crossover_probability = cfg["crossover"]["min_probability"]
                self.max_crossover_probability = cfg["crossover"]["max_probability"]

            elif self.crossover_strategy == CrossoverStrategy.ADAPTIVE_GLOBAL:
                self.crossover_probability = cfg["crossover"]["init_probability"]
                self.min_crossover_probability = cfg["crossover"]["min_probability"]
                self.max_crossover_probability = cfg["crossover"]["max_probability"]
                self.crossover_inc_factor = cfg["crossover"]["increase_factor"]
                self.crossover_dec_factor = cfg["crossover"]["decrease_factor"]
