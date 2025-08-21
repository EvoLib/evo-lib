# SPDX-License-Identifier: MIT
"""
EvoControlParams is a simple container class that holds runtime mutation and crossover
parameters used in evolutionary strategies.

It is representation-agnostic and does not perform any updates or behavior.
All update logic is handled externally by `evo_param_update.py`.
"""

import numpy as np

from evolib.interfaces.enums import CrossoverStrategy, MutationStrategy
from evolib.interfaces.types import CrossoverFunction


class EvoControlParams:
    """
    Container for all evolution-related control parameters.

    Stores mutation and crossover configuration for use during evolutionary runs. Can be
    updated externally by passing to `evo_param_update.update_*()` functions.
    """

    def __init__(self) -> None:

        # Mutationstrategy
        self.mutation_strategy: MutationStrategy | None = None

        # Global Mutationparameter
        self.mutation_strength: float | None = None
        self.mutation_probability: float | None = None
        self.tau: float = 0.0

        # Per Paramter mutation
        self.mutation_strengths: np.ndarray | None = None

        # Bounds for mutation (min/max)
        self.min_mutation_strength: float | None = None
        self.max_mutation_strength: float | None = None
        self.min_mutation_probability: float | None = None
        self.max_mutation_probability: float | None = None

        # Diversity based Adaptionfaktors
        self.mutation_inc_factor: float | None = None
        self.mutation_dec_factor: float | None = None
        self.min_diversity_threshold: float | None = None
        self.max_diversity_threshold: float | None = None

        # Crossover
        self.crossover_strategy: CrossoverStrategy | None = None
        self.crossover_probability: float | None = None
        self.min_crossover_probability: float | None = None
        self.max_crossover_probability: float | None = None
        self.crossover_inc_factor: float | None = None
        self.crossover_dec_factor: float | None = None
        self._crossover_fn: CrossoverFunction | None = None

    def get_log_dict(self) -> dict[str, float]:
        """
        Returns a flat dictionary of key parameters, useful for tracking or logging.

        Returns:
            dict[str, float]: Dictionary with mutation/crossover values
        """
        return {
            "mutation_strength": self.mutation_strength or 0.0,
            "mutation_probability": self.mutation_probability or 0.0,
            "tau": self.tau or 0.0,
            "crossover_probability": self.crossover_probability or 0.0,
        }
