# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Union

import numpy as np

if TYPE_CHECKING:
    from evolib.interfaces.types import ModuleConfig


class ParaBase(ABC):
    """
    Abstract base class for all evolvable parameter representations (e.g. Vector,
    EvoNet).

    This interface defines the evolutionary behavior of individuals and allows mutation,
    crossover, adaptive updates, and access to history and diagnostic information.
    """

    def __init__(self) -> None:
        self._crossover_fn: (
            Callable[
                [np.ndarray, np.ndarray],
                Union[np.ndarray, tuple[np.ndarray, np.ndarray]],
            ]
            | None
        ) = None

    @abstractmethod
    def apply_config(self, cfg: "ModuleConfig") -> None:
        """Initializes parameters from a configuration object."""
        ...

    @abstractmethod
    def mutate(self) -> None:
        """Applies mutation to the parameters."""
        ...

    @abstractmethod
    def print_status(self) -> None: ...

    @abstractmethod
    def get_status(self) -> str | dict[str, Any]: ...

    def get_history(self) -> dict[str, float]:
        """Returns a dictionary of scalar values for logging (optional)."""
        return {}

    def update_mutation_parameters(
        self, generation: int, max_generations: int, diversity_ema: float | None = None
    ) -> None:
        """
        Optional: Override in subclasses that support strategy-dependent
        mutation control.
        """
        pass

    def update_crossover_parameters(
        self, generation: int, max_generations: int, diversity_ema: float | None = None
    ) -> None:
        """
        Optional: Override in subclasses that support strategy-dependent
        crossover control.
        """
        pass

    def crossover_with(self, partner: "ParaBase") -> None:
        """
        Delegates crossover logic to the assigned strategy function.

        Must be overridden if internal application logic differs.
        """
        if self._crossover_fn is None:
            raise RuntimeError(
                f"No crossover function defined for {self.__class__.__name__}"
            )
        raise NotImplementedError(
            "crossover_with must be implemented in the concrete Para subclass "
            "to interpret the result of _crossover_fn()."
        )
