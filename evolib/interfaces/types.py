# SPDX-License-Identifier: MIT

from typing import TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    from evolib.core.population import Pop  # noqa: F401
    from evolib.core.individual import Indiv  # noqa: F401

from typing import Callable, Optional, Protocol

from evolib.config.evonet_component_config import EvoNetComponentConfig
from evolib.config.vector_component_config import VectorComponentConfig
from evolib.interfaces.structs import MutationParams
from evolib.representation.base import ParaBase

EvolutionStrategyFunction = Callable[["Pop"], None]
SelectionFunction = Callable[["Pop", int], list["Indiv"]]
ReplaceFunction = Callable[["Pop", list["Indiv"]], None]
CrossoverFunction = Callable[
    [np.ndarray, np.ndarray], np.ndarray | tuple[np.ndarray, np.ndarray]
]
ParaInitializer = Callable[["Pop"], ParaBase]

ModuleConfig = Union[VectorComponentConfig, EvoNetComponentConfig]

# Base type for all population-related hooks
PopulationHook = Callable[["Pop"], None]

# Specific hook aliases for clarity in Pop.run
OnStartHook = Optional[PopulationHook]
OnGenerationHook = Optional[PopulationHook]
OnImprovementHook = Optional[PopulationHook]
OnEndHook = Optional[PopulationHook]


class FitnessFunction(Protocol):
    def __call__(self, indiv: "Indiv") -> None: ...


class MutationFunction(Protocol):
    def __call__(self, indiv: "Indiv", params: MutationParams) -> None: ...


class TauUpdateFunction(Protocol):
    def __call__(self, indiv: "Indiv") -> None: ...


ParaInitFunction = Callable[["Pop"], ParaBase]
