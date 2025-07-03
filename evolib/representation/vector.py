import numpy as np

from evolib.interfaces.structs import MutationParams
from evolib.representation.base import ParaBase
from evolib.operators.mutation import adapted_mutation_strength


class ParaVector(ParaBase):
    def __init__(self) -> None:

        self.mutation_strategy = None
        self.mutation_strength = None
        self.mutation_probability = None
        self.tau = 0.0
        self.para_mutation_strengths = None
        self.bounds = None

        self.vector = np.zeros(1)
        self.min_mutation_strength = None
        self.max_mutation_strength = None
        self.min_mutation_probability = None
        self.max_mutation_probability = None
        self.mutation_inc_factor = None
        self.mutation_dec_factor = None
        self.min_diversity_threshold = None
        self.max_diversity_threshold = None

        self.crossover_strategy = None
        self.crossover_probability = None
        self.crossover_inc_factor = None
        self.crossover_dec_factor = None

    def mutate(self) -> None:
        """
        Applies Gaussian mutation to the parameter vector.

        If self.para_mutation_strengths is defined, gene-specific strengths are used.
        Otherwise, the global mutation strength from params.strength is applied.
        """
        if self.para_mutation_strengths is not None:
            # Adaptive per-parameter
            self.vector += np.random.normal(
                loc=0.0, scale=self.para_mutation_strengths, size=len(self.vector)
            )
        else:
            noise = np.random.normal(
                loc=0.0, scale=self.mutation_strength, size=len(self.vector)
            )
            mask = np.random.rand(len(self.vector)) < self.mutation_probability
            self.vector += noise * mask

        self.vector = np.clip(self.vector, self.bounds)

    def update_tau(self) -> None:
        """
        Update the learning rate tau based on the vector length.

        This implements a simple self-adaptation rule:
        tau = 1 / sqrt(n), where n = number of parameters.
        """
        if self.tau is not None and hasattr(self, "__len__"):
            n = len(self)
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

        self.mutation_strength = adapted_strength(params)


    def adapt_para_mutation_strengths(self, params: MutationParams) -> None:
        """
        Adapt each gene-specific mutation strength using independent log-normal factors.

        This corresponds to the self-adaptive mutation strategy where each parameter
        dimension maintains its own mutation strength (sigma_i).
        """
        self.para_mutation_strengths *= np.exp(
            self.tau * np.random.normal(size=len(self.vector))
        )
        self.para_mutation_strengths = np.clip(
            self.para_mutation_strengths, params.min_strength, params.max_strength
        )

    def copy(self) -> "ParaVector":
        return ParaVector(
            vector=self.vector.copy(),
            tau=self.tau,
            para_mutation_strengths=self.para_mutation_strengths.copy(),
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

    def update_mutation_parameters(self, generation: int) -> None:
        """Update mutation parameters based on strategy and generation."""
        if self.mutation_strategy == MutationStrategy.EXPONENTIAL_DECAY:
            self.mutation_strength = _exponential_mutation_strength(generation)
            self.mutation_probability = _exponential_mutation_probability(generation)

        elif self.mutation_strategy == MutationStrategy.ADAPTIVE_GLOBAL:
            self.mutation_probability = 0 #TODO
            self.mutation_strength = 0 #TODO


    def _exponential_mutation_strength(self, generation: int) -> float:
        """
        Calculates exponentially decaying mutation strength over generations.

        Args:
            generation: int: 

        Returns:
            float: The adjusted mutation strength.
        """
        k = (
            np.log(self.max_mutation_strength / self.min_mutation_strength)
            / self.max_generations
        )
        return self.max_mutation_strength * np.exp(-k * generation)

    def _exponential_mutation_probability(generation: int) -> float:
        """
        Calculates exponentially decaying mutation probablility over generations.

        Args:
            generation: int

        Returns:
            float: The adjusted mutation rate.
        """
        k = (
            np.log(self.max_mutation_probability / self.min_mutation_probability)
            / self.max_generations
        )
        return self.max_mutation_probability * np.exp(-k * self.generation_num)
