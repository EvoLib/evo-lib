# SPDX-License-Identifier: MIT
import math
from typing import TYPE_CHECKING, List

from evolib.interfaces.enums import OptimizationDirection

if TYPE_CHECKING:
    from evolib.core.individual import Indiv


def sort_by_fitness(
    indivs: List["Indiv"], optimization_direction: OptimizationDirection
) -> List["Indiv"]:
    """
    Sort individuals with the best fitness first.

    Unevaluated or non-finite fitness values are always treated as worst.

    Args:
        indivs: Individuals to sort.
        optimization_direction: Whether lower or higher fitness values are better.

    Returns:
        Sorted list of individuals.
    """
    maximize = optimization_direction == OptimizationDirection.MAXIMIZE

    def fitness_key(ind: "Indiv") -> float:
        if ind.fitness is None or not math.isfinite(ind.fitness):
            return float("-inf") if maximize else float("inf")
        return ind.fitness

    return sorted(indivs, key=fitness_key, reverse=maximize)
