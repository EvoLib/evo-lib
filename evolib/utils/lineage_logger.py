# SPDX-License-Identifier: MIT
"""
lineage_logger.py — Optional micro-level lineage logging.

This logger writes one record per individual and generation, tracking ancestry,
structural events, and HELI involvement.
"""

from __future__ import annotations

import csv
import os

from evolib.core.individual import Indiv


class LineageLogger:
    """
    Lightweight CSV-based lineage logger.

    It records per-individual information per generation,
    enabling survival-time and innovation-retention analyses.

    Parameters
    ----------
    path : str
        Target file path for CSV output.
    """

    def __init__(self, path: str = "lineage_log.csv") -> None:
        self.path = path
        self._initialized = False

    def _ensure_header(self) -> None:
        """Create file and write header if not present."""
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "generation",
                        "indiv_id",
                        "parent_id",
                        "birth_gen",
                        "is_elite",
                        "fitness",
                        "age",
                        "is_structural_mutant",
                        "heli_seed",
                        "heli_reintegrated",
                    ]
                )
            self._initialized = True

    def log_generation(self, indivs: list[Indiv], generation: int) -> None:
        """
        Append lineage info for a generation.

        Parameters
        ----------
        indivs : list[Indiv]
            List of individuals in the current population.
        generation : int
            Current generation index.
        """
        if not self._initialized:
            self._ensure_header()

        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            for indiv in indivs:
                writer.writerow(
                    [
                        generation,
                        getattr(indiv, "id", ""),
                        getattr(indiv, "parent_id", ""),
                        getattr(indiv, "birth_gen", 0),
                        int(getattr(indiv, "is_elite", False)),
                        getattr(indiv, "fitness", None),
                        getattr(indiv, "age", 0),
                        int(getattr(indiv, "is_structural_mutant", False)),
                        int(getattr(indiv, "heli_seed", False)),
                        int(getattr(indiv, "heli_reintegrated", False)),
                    ]
                )
