# SPDX-License-Identifier: MIT
"""I/O and visualization support for the Predator-Prey simulation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from evosim.sims.predator_prey.simulation import PredatorPreySimulation

if TYPE_CHECKING:
    from evosim.renderers.pygame_common import PygameWindow
    from evosim.renderers.pygame_predator_prey import PygamePredatorPreyRenderer


class PredatorPreySession:
    """Handle metrics output and optional visualization for one simulation."""

    def __init__(
        self,
        simulation: PredatorPreySimulation,
        *,
        render: bool = False,
    ) -> None:
        self.simulation = simulation
        self._closed = False
        self._window: PygameWindow | None = None
        self._renderer: PygamePredatorPreyRenderer | None = None
        self._file: TextIO | None = None
        self._writer: csv.DictWriter[str] | None = None
        self._last_reported_step: int | None = None

        metrics_file = self.simulation.config.metrics.file
        if metrics_file is not None:
            metrics_path = Path(metrics_file)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = metrics_path.open("w", encoding="utf-8", newline="")
            metrics = simulation.metrics()
            self._writer = csv.DictWriter(self._file, fieldnames=list(metrics))
            self._writer.writeheader()

        if render:
            self._open_renderer()

        self._report_metrics()
        if not simulation.running:
            self.close()

    @property
    def running(self) -> bool:
        """Return whether the session is still active."""
        return not self._closed

    def update(self) -> None:
        """Update metrics output and optional visualization."""
        if self._closed:
            return

        self._process_events()
        if self._closed:
            return

        if self.simulation.step_count % self.simulation.config.metrics.interval == 0:
            self._report_metrics()

        if self._window is not None and self._renderer is not None:
            self._renderer.draw(
                self._window.screen,
                self.simulation,
                self._window.font,
            )
            self._window.update()

        if not self.simulation.running:
            self.close()

    def close(self) -> None:
        """Report final metrics and release session resources."""
        if self._closed:
            return

        if self._last_reported_step != self.simulation.step_count:
            self._report_metrics()

        if self._window is not None:
            self._window.close()
        if self._file is not None:
            self._file.close()
        self._closed = True

    def _process_events(self) -> None:
        if self._window is None:
            return

        import pygame

        pressed = self._window.process_events()
        if pygame.K_s in pressed and self._renderer is not None:
            self._renderer.toggle_sensors()
        if not self._window.running:
            self.close()

    def _open_renderer(self) -> None:
        from evosim.renderers.pygame_common import DEFAULT_FPS, PygameWindow
        from evosim.renderers.pygame_predator_prey import PygamePredatorPreyRenderer

        self._renderer = PygamePredatorPreyRenderer()
        self._window = PygameWindow(
            self.simulation.world_size,
            caption="EvoSim - Predator-Prey",
            fps=DEFAULT_FPS,
            panel_width=self._renderer.panel_width,
        )

    def _report_metrics(self) -> None:
        metrics = self.simulation.metrics()

        if self._writer is not None and self._file is not None:
            self._writer.writerow(metrics)
            self._file.flush()

        self._last_reported_step = self.simulation.step_count
        print(
            f"step={metrics['step']} "
            f"prey={metrics['prey_population']} "
            f"predators={metrics['predator_population']} "
            f"captured={metrics['prey_captured']} "
            f"prey_births={metrics['prey_births']} "
            f"predator_births={metrics['predator_births']}"
        )
