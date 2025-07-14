# SPDX-License-Identifier: MIT
"""
Maps selection strategy identifiers to their corresponding selection functions.

This registry enables dynamic selection of parent selection strategies by referencing
symbolic identifiers defined in the `SelectionStrategy` enum.

Usage:
    selected = selection_registry[SelectionStrategy.TOURNAMENT](pop, num_parents)

Each function returns a list of selected parents (deep copies).
"""

from functools import partial

from evolib.config.schema import SelectionConfig
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


def build_selection_registry(
    cfg: SelectionConfig,
) -> dict[SelectionStrategy, SelectionFunction]:
    """Create a selection registry with strategy-specific parameters bound from
    config."""
    return {
        SelectionStrategy.TOURNAMENT: partial(
            selection_tournament,
            tournament_size=cfg.tournament_size or 3,
            fitness_maximization=cfg.fitness_maximization or False,
        ),
        SelectionStrategy.ROULETTE: partial(
            selection_roulette,
            fitness_maximization=cfg.fitness_maximization or False,
        ),
        SelectionStrategy.RANK_LINEAR: partial(
            selection_rank_based,
            mode="linear",
            fitness_maximization=cfg.fitness_maximization or False,
        ),
        SelectionStrategy.RANK_EXPONENTIAL: partial(
            selection_rank_based,
            mode="exponential",
            exp_base=cfg.exp_base or 2.0,
            fitness_maximization=cfg.fitness_maximization or False,
        ),
        SelectionStrategy.SUS: partial(
            selection_sus,
            fitness_maximization=cfg.fitness_maximization or False,
        ),
        SelectionStrategy.BOLTZMANN: partial(
            selection_boltzmann,
            temperature=cfg.exp_base or 1.0,
            fitness_maximization=cfg.fitness_maximization or False,
        ),
        SelectionStrategy.TRUNCATION: partial(
            selection_truncation,
            fitness_maximization=cfg.fitness_maximization or False,
        ),
        SelectionStrategy.RANDOM: lambda pop, n: selection_random(pop)[:n],
    }
