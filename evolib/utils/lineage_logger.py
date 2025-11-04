# SPDX-License-Identifier: MIT
"""
Event-based lineage logging.

This logger records evolutionary events, it is designed for detailed analysis of
individual lifecycles, including birth, survival, death, structural mutations and HELI-
related transitions.

The output CSV file can be used directly with Pandas, Plotly, or other visual analytics
tools to reconstruct full lineage graphs.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from evolib.population import Indiv


class LineageLogger:
    """Event-based lineage logger for evolutionary analysis."""

    def __init__(self, filename: Union[str, Path]) -> None:
        self.filename: Path = filename if isinstance(filename, Path) else Path(filename)
        self._init_csv()

    def _init_csv(self) -> None:
        """Initialize the CSV file with header columns."""
        with self.filename.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "generation",
                    "indiv_id",
                    "parent_id",
                    "event",
                    "birth_gen",
                    "exit_gen",
                    "is_elite",
                    "is_structural_mutant",
                    "heli_reintegrated",
                    "notes",
                ]
            )

    def log_event(
        self,
        indiv: Indiv,
        generation: int,
        event: str,
        note: str = "",
    ) -> None:
        """Log a single evolutionary event."""
        with self.filename.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    generation,
                    getattr(indiv, "id", None),
                    getattr(indiv, "parent_id", None),
                    event,
                    getattr(indiv, "birth_gen", None),
                    getattr(indiv, "exit_gen", None),
                    int(getattr(indiv, "is_elite", False)),
                    int(getattr(indiv, "is_structural_mutant", False)),
                    int(getattr(indiv, "heli_reintegrated", False)),
                    note,
                ]
            )

    def log_population(
        self,
        indivs: list[Indiv],
        generation: int,
        event: str,
        note: str = "",
    ) -> None:
        """Log the same event for multiple individuals."""
        if not indivs:
            return
        with self.filename.open("a", newline="") as f:
            writer = csv.writer(f)
            for indiv in indivs:
                writer.writerow(
                    [
                        generation,
                        getattr(indiv, "id", None),
                        getattr(indiv, "parent_id", None),
                        event,
                        getattr(indiv, "birth_gen", None),
                        getattr(indiv, "exit_gen", None),
                        int(getattr(indiv, "is_elite", False)),
                        int(getattr(indiv, "is_structural_mutant", False)),
                        int(getattr(indiv, "heli_reintegrated", False)),
                        note,
                    ]
                )
