from abc import ABC, abstractmethod


class ParaBase(ABC):
    @abstractmethod
    def apply_config(self, cfg: dict) -> None: ...

    @abstractmethod
    def mutate(self) -> None: ...

    @abstractmethod
    def print_status(self) -> None: ...

    @abstractmethod
    def get_status(self) -> str: ...

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
