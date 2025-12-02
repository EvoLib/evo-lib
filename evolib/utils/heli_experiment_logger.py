# SPDX-License-Identifier: MIT

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Type

import numpy as np

if TYPE_CHECKING:
    from evolib.core.population import Pop


class HeliExperimentLogger:
    """
    Lightweight experiment-level logger for HELI-specific population metrics.

    Logs generation-wise aggregates about population structure (weights / neurons)
    """

    def __init__(self, filename: str | Path):
        self.filename = Path(filename)
        self._file = open(self.filename, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            [
                "generation",
                "mean_num_weights",
                "mean_num_neurons",
                "base_fitness_evaluations_gen",
                "heli_fitness_evaluations_gen",
                "heli_overhead_percent_gen",
                "fitness_evaluations_total",
                "heli_fitness_evaluations_total",
            ]
        )
        self._file.flush()

    def log_generation(self, pop: Pop) -> None:
        """
        Collects and logs HELI-relevant metrics for one generation.

        generation (int): Current generation number.
        indivs (list[Indiv]): Population individuals Expected to have `.para.net`
        """

        heli_gen = pop.heli_fitness_evaluations_gen
        base_gen = pop.parent_pool_size + pop.offspring_pool_size
        overhead_pct_gen = (heli_gen / base_gen * 100.0) if base_gen > 0 else 0.0

        weights, neurons = [], []

        for indiv in pop.indivs:
            net = indiv.para["brain"].net
            weights.append(net.num_weights)
            neurons.append(net.num_hidden)

        mean_num_weights = np.mean([weight for weight in weights])
        mean_num_neurons = np.mean([neuron for neuron in neurons])

        self._writer.writerow(
            [
                pop.generation_num,
                mean_num_weights,
                mean_num_neurons,
                base_gen,
                heli_gen,
                overhead_pct_gen,
                pop.fitness_evaluations_total,
                pop.heli_fitness_evaluations_total,
            ]
        )
        self._file.flush()

    def close(self) -> None:
        """Close the CSV file cleanly."""
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> HeliExperimentLogger:
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> None:
        self.close()
