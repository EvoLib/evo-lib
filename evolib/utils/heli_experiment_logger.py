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

    def __init__(
        self,
        filename: str | Path,
        heli_lineage_filename: str | Path | None = None,
    ):
        self.filename = Path(filename)
        self._file = open(self.filename, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            [
                "generation",
                "mean_num_weights",
                "mean_num_neurons",
                "structural_mutants_gen",
                "heli_seeds_gen",
                "base_fitness_evaluations_gen",
                "heli_fitness_evaluations_gen",
                "heli_overhead_percent_gen",
                "fitness_evaluations_total",
                "heli_fitness_evaluations_total",
                "total_fitness_evaluations",
            ]
        )
        self._file.flush()
        self._last_fitness_evaluations_total = 0
        self._heli_lineage_file = None
        self._heli_lineage_writer = None

        if heli_lineage_filename is not None:
            lineage_path = Path(heli_lineage_filename)
            self._heli_lineage_file = open(lineage_path, "w", newline="")
            self._heli_lineage_writer = csv.writer(self._heli_lineage_file)
            self._heli_lineage_writer.writerow(
                [
                    "generation",
                    "seed_id",
                    "seed_fitness",
                    "result_id",
                    "result_fitness",
                    "fitness_improvement",
                    "incubation_generations",
                    "incubation_evaluations",
                ]
            )
            self._heli_lineage_file.flush()

    def log_generation(self, pop: Pop) -> None:
        """
        Collects and logs HELI-relevant metrics for one generation.

        generation (int): Current generation number. indivs (list[Indiv]): Population
        individuals Expected to have `.para.net`
        """
        heli_gen = pop.heli_fitness_evaluations_gen
        base_gen = pop.fitness_evaluations_total - self._last_fitness_evaluations_total
        self._last_fitness_evaluations_total = pop.fitness_evaluations_total
        overhead_pct_gen = (heli_gen / base_gen * 100.0) if base_gen > 0 else 0.0
        total_evaluations = (
            pop.fitness_evaluations_total + pop.heli_fitness_evaluations_total
        )

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
                pop.structural_mutants_gen,
                pop.heli_seeds_gen,
                base_gen,
                heli_gen,
                overhead_pct_gen,
                pop.fitness_evaluations_total,
                pop.heli_fitness_evaluations_total,
                total_evaluations,
            ]
        )
        self._file.flush()

        if self._heli_lineage_writer is not None:
            for record in pop.heli_lineage_records_gen:
                self._heli_lineage_writer.writerow(
                    [
                        pop.generation_num,
                        record["seed_id"],
                        record["seed_fitness"],
                        record["result_id"],
                        record["result_fitness"],
                        record["fitness_improvement"],
                        record["incubation_generations"],
                        record["incubation_evaluations"],
                    ]
                )

            if self._heli_lineage_file is not None:
                self._heli_lineage_file.flush()

    def close(self) -> None:
        """Close the CSV file cleanly."""
        if not self._file.closed:
            self._file.close()
        if self._heli_lineage_file is not None and not self._heli_lineage_file.closed:
            self._heli_lineage_file.close()

    def __enter__(self) -> HeliExperimentLogger:
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> None:
        self.close()
