from abc import ABC, abstractmethod

from evolib.interfaces.types import MutationParams


class ParaBase(ABC):
    @abstractmethod
    def mutate(self, params: MutationParams) -> None: ...

    @abstractmethod
    def print_status(self) -> None: ...

    @abstractmethod
    def get_status(self) -> str: ...
