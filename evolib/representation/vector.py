import numpy as np

from evolib.interfaces.structs import MutationParams
from evolib.representation.base import ParaBase
from evolib.operators.mutation import adapt_mutation_strength


class ParaVector(ParaBase):
    def __init__(
        self,
        vector: np.ndarray,
        mutation_strength: float | None = None,
        mutation_probability: float | None = None,
        tau: float = 0.0,
        para_mutation_strengths: np.ndarray | None = None,
    ):

        self.vector = vector
        self.mutation_strength = mutation_strength
        self.mutation_probability = mutation_probability
        self.tau = tau
        self.para_mutation_strengths = (
            para_mutation_strengths
            if para_mutation_strengths is not None
            else np.full(len(vector), 0.1)  # Defaultwert kann angepasst werden
        )

    def mutate(self, params: MutationParams) -> None:
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
                loc=0.0, scale=params.strength, size=len(self.vector)
            )
            mask = np.random.rand(len(self.vector)) < self.mutation_probability
            self.vector += noise * mask

        self.vector = np.clip(self.vector, *params.bounds)

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
