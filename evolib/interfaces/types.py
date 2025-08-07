# SPDX-License-Identifier: MIT

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from evolib.core.population import Pop  # noqa: F401
    from evolib.core.individual import Indiv  # noqa: F401

from collections.abc import Callable
from typing import Protocol

from evolib.config.evonet_component_config import EvoNetComponentConfig
from evolib.config.vector_component_config import VectorComponentConfig
from evolib.interfaces.structs import MutationParams
from evolib.representation.base import ParaBase

EvolutionStrategyFunction = Callable[["Pop"], None]
SelectionFunction = Callable[["Pop", int], list["Indiv"]]
ReplaceFunction = Callable[["Pop", list["Indiv"]], None]
ParaInitializer = Callable[["Pop"], ParaBase]

ModuleConfig = Union[VectorComponentConfig, EvoNetComponentConfig]


class FitnessFunction(Protocol):
    def __call__(self, indiv: "Indiv") -> None: ...


class MutationFunction(Protocol):
    def __call__(self, indiv: "Indiv", params: MutationParams) -> None: ...


class TauUpdateFunction(Protocol):
    def __call__(self, indiv: "Indiv") -> None: ...


ParaInitFunction = Callable[["Pop"], ParaBase]


class CrossoverFunction(Protocol):
    def __call__(self, parent1: "Indiv", parent2: "Indiv") -> list["Indiv"]: ...
