# SPDX-License-Identifier: MIT
from typing import List

from evolib.core.population import Indiv


def sort_by_fitness(indivs: List[Indiv], maximize: bool = False) -> List[Indiv]:
    """
    Sorts individuals by fitness.

    Unevaluated individuals (fitness is None) are treated as worst:
    - For minimization: +inf (they will end up at the end)
    - For maximization: -inf (they will end up at the end)

    Args:
        indivs: List of individuals to sort.
        maximize: If True, sort descending (higher fitness is better).
                  If False, sort ascending (lower fitness is better).

    Returns:
        Sorted list of individuals.
    """

    def fitness_key(ind: "Indiv") -> float:
        if ind.fitness is None:
            return float("-inf") if maximize else float("inf")
        return ind.fitness

    return sorted(indivs, key=fitness_key, reverse=maximize)
