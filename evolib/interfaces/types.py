# SPDX-License-Identifier: MIT

from typing import Any, Protocol

from evolib.core.individual import Indiv
from evolib.interfaces.structs import MutationParams


class FitnessFunction(Protocol):
    def __call__(self, indiv: Indiv) -> None: ...


class MutationFunction(Protocol):
    def __call__(self, indiv: Indiv, params: MutationParams) -> None: ...


class TauUpdateFunction(Protocol):
    def __call__(self, indiv: Indiv) -> None: ...


class ParaInitializer(Protocol):
    def __call__(self) -> Any: ...


class CrossoverFunction(Protocol):
    def __call__(self, parent1: Indiv, parent2: Indiv) -> list[Indiv]: ...
