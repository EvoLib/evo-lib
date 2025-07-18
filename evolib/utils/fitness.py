# SPDX-License-Identifier: MIT
from typing import List

from evolib.core.population import Indiv


def sort_by_fitness(indivs: List[Indiv], maximize: bool = False) -> List[Indiv]:
    """
    Sorts individuals by fitness.

    Args:
        indivs: List of individuals to sort.
        maximize: If True, sort descending (higher fitness is better).
                  If False, sort ascending (lower fitness is better).

    Returns:
        Sorted list of individuals.
    """
    return sorted(indivs, key=lambda i: i.fitness, reverse=maximize)
