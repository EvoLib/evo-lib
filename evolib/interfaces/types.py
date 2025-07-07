# SPDX-License-Identifier: MIT

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evolib.core.population import Pop  # noqa: F401

from collections.abc import Callable
from typing import Protocol

from evolib.core.individual import Indiv
from evolib.interfaces.structs import MutationParams
from evolib.representation.base import ParaBase


class FitnessFunction(Protocol):
    def __call__(self, indiv: Indiv) -> None: ...


class MutationFunction(Protocol):
    def __call__(self, indiv: Indiv, params: MutationParams) -> None: ...


class TauUpdateFunction(Protocol):
    def __call__(self, indiv: Indiv) -> None: ...


ParaInitFunction = Callable[["Pop"], ParaBase]


class ParaInitializer(Protocol):
    def __call__(self, cfg: dict) -> ParaInitFunction: ...


class CrossoverFunction(Protocol):
    def __call__(self, parent1: Indiv, parent2: Indiv) -> list[Indiv]: ...
