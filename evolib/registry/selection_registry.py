# SPDX-License-Identifier: MIT
"""
Maps selection strategy identifiers to their corresponding selection functions.

This registry enables dynamic selection of parent selection strategies by referencing
symbolic identifiers defined in the `SelectionStrategy` enum.

Usage:
    selected = selection_registry[SelectionStrategy.TOURNAMENT](pop, num_parents)

Each function returns a list of selected parents (deep copies).
"""

from evolib.interfaces.enums import SelectionStrategy
from evolib.interfaces.types import SelectionFunction
from evolib.operators.selection import (
    selection_boltzmann,
    selection_random,
    selection_rank_based,
    selection_roulette,
    selection_sus,
    selection_tournament,
    selection_truncation,
)

selection_registry: dict[SelectionStrategy, SelectionFunction] = {
    SelectionStrategy.TOURNAMENT: selection_tournament,
    SelectionStrategy.ROULETTE: selection_roulette,
    SelectionStrategy.RANK_LINEAR: lambda pop, n: selection_rank_based(
        pop, num_parents=n, mode="linear"
    ),
    SelectionStrategy.RANK_EXPONENTIAL: lambda pop, n: selection_rank_based(
        pop, num_parents=n, mode="exponential"
    ),
    SelectionStrategy.SUS: selection_sus,
    SelectionStrategy.BOLTZMANN: selection_boltzmann,
    SelectionStrategy.TRUNCATION: selection_truncation,
    SelectionStrategy.RANDOM: lambda pop, n: selection_random(pop)[:n],
}
