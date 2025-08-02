# SPDX-License-Identifier: MIT
"""
EvoParamControl provides centralized management of mutation and crossover parameters
used in evolutionary strategies.

It handles the configuration and per-generation update logic for adaptive and
non-adaptive mutation/crossover strategies, independent of the actual parameter
representation (e.g. vector, network, tensor).

Note: EvoParamControl does not perform mutation or crossover operations directly,
but supplies up-to-date parameters for those operations.
"""

import numpy as np

from evolib.interfaces.enums import CrossoverStrategy, MutationStrategy


class EvoParamControl:
    """
    Controls runtime adaptation of mutation and crossover parameters.

    This class encapsulates mutation and crossover strategies such as:
    - Constant
    - Exponential decay
    - Diversity-adaptive control

    Responsibilities:
    - Store and update global mutation/crossover parameters
    - Apply per-generation adjustments
    - Remain agnostic to representation (no direct mutation/crossover)

    Usage:
        control = EvoParamControl()
        control.update_mutation_parameters(generation=5, ...)
        sigma = control.mutation_strength
    """

    def __init__(self) -> None:
        # Mutation
        self.mutation_strategy: MutationStrategy | None = None
        self.mutation_strength: float | None = None
        self.mutation_probability: float | None = None
        self.tau: float = 0.0

        self.min_mutation_strength: float | None = None
        self.max_mutation_strength: float | None = None
        self.min_mutation_probability: float | None = None
        self.max_mutation_probability: float | None = None

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

    # Mutation Update

    def update_mutation_parameters(
        self,
        generation: int,
        max_generations: int,
        diversity_ema: float | None = None,
    ) -> None:
        """
        Update mutation strength and probability based on current strategy.

        Args:
            generation (int): Current generation number
            max_generations (int): Max number of generations
            diversity_ema (float, optional): Exponential moving average of
            population diversity
        """
        match self.mutation_strategy:
            case MutationStrategy.EXPONENTIAL_DECAY:
                self.mutation_strength = self._exp_decay(
                    generation,
                    max_generations,
                    self.max_mutation_strength,
                    self.min_mutation_strength,
                )
                self.mutation_probability = self._exp_decay(
                    generation,
                    max_generations,
                    self.max_mutation_probability,
                    self.min_mutation_probability,
                )

            case MutationStrategy.ADAPTIVE_GLOBAL:
                if diversity_ema is None:
                    raise ValueError("diversity_ema required for ADAPTIVE_GLOBAL")
                self.mutation_strength = self._adaptive_value(
                    value=self.mutation_strength,
                    min_value=self.min_mutation_strength,
                    max_value=self.max_mutation_strength,
                    diversity_ema=diversity_ema,
                )
                self.mutation_probability = self._adaptive_value(
                    value=self.mutation_probability,
                    min_value=self.min_mutation_probability,
                    max_value=self.max_mutation_probability,
                    diversity_ema=diversity_ema,
                )

    # Crossover Update

    def update_crossover_parameters(
        self,
        generation: int,
        max_generations: int,
        diversity_ema: float | None = None,
    ) -> None:
        """
        Update crossover probability based on strategy and diversity.

        Args:
            generation (int): Current generation
            max_generations (int): Maximum number of generations
            diversity_ema (float, optional): EMA of population diversity
        """
        match self.crossover_strategy:
            case CrossoverStrategy.EXPONENTIAL_DECAY:
                self.crossover_probability = self._exp_decay(
                    generation,
                    max_generations,
                    self.max_crossover_probability,
                    self.min_crossover_probability,
                )

            case CrossoverStrategy.ADAPTIVE_GLOBAL:
                if diversity_ema is None:
                    raise ValueError("diversity_ema required for ADAPTIVE_GLOBAL")
                self.crossover_probability = self._adaptive_value(
                    value=self.crossover_probability,
                    min_value=self.min_crossover_probability,
                    max_value=self.max_crossover_probability,
                    diversity_ema=diversity_ema,
                )

    # Internal Helpers

    def _exp_decay(
        self,
        generation: int,
        max_generations: int,
        max_value: float | None,
        min_value: float | None,
    ) -> float | None:
        """Returns exponentially decayed value or None if inputs are incomplete."""
        if max_value is None or min_value is None:
            return None
        if generation > max_generations:
            return min_value
        k = np.log(max_value / min_value) / max_generations
        return max_value * np.exp(-k * generation)

    def _adaptive_value(
        self,
        value: float | None,
        min_value: float | None,
        max_value: float | None,
        diversity_ema: float,
    ) -> float | None:
        """Adjusts value adaptively based on diversity thresholds."""
        if (
            value is None
            or min_value is None
            or max_value is None
            or self.mutation_inc_factor is None
            or self.mutation_dec_factor is None
            or self.min_diversity_threshold is None
            or self.max_diversity_threshold is None
        ):
            return value

        if diversity_ema < self.min_diversity_threshold:
            return min(max_value, value * self.mutation_inc_factor)
        elif diversity_ema > self.max_diversity_threshold:
            return max(min_value, value * self.mutation_dec_factor)
        return value

    def get_log_dict(self) -> dict[str, float]:
        """Returns a dictionary of currently active evolution parameters."""
        return {
            "mutation_strength": self.mutation_strength or 0.0,
            "mutation_probability": self.mutation_probability or 0.0,
            "crossover_probability": self.crossover_probability or 0.0,
            "tau": self.tau or 0.0,
        }
