# SPDX-License-Identifier: MIT
"""I/O and visualization support for the Foraging simulation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from evosim.sims.foraging.simulation import ForagingSimulation

if TYPE_CHECKING:
    from evosim.renderers.pygame_common import PygameWindow
    from evosim.renderers.pygame_foraging import PygameForagingRenderer


class ForagingSession:
    """Handle metrics output and optional visualization for one simulation."""

    def __init__(
        self,
        simulation: ForagingSimulation,
        *,
        render: bool = False,
        output_dir: str | Path = ".",
    ) -> None:
        self.simulation = simulation
        self._closed = False
        self._window: PygameWindow | None = None
        self._renderer: PygameForagingRenderer | None = None

        metrics_path = self._metrics_path(Path(output_dir))
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO = metrics_path.open("w", encoding="utf-8", newline="")

        metrics = simulation.metrics()
        self._writer = csv.DictWriter(self._file, fieldnames=list(metrics))
        self._writer.writeheader()
        self._last_logged_step: int | None = None

        if render:
            self._open_renderer()

        self._write_metrics()
        if not simulation.running:
            self.close()

    @property
    def running(self) -> bool:
        """Return whether the session is still active."""
        return not self._closed

    def update(self) -> None:
        """Update session state, metrics output, and optional visualization."""
        if self._closed:
            return

        self._process_events()
        if self._closed:
            return

        if self.simulation.step_count % self.simulation.config.metrics.interval == 0:
            self._write_metrics()

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
        """Write final metrics and release session resources."""
        if self._closed:
            return

        if self._last_logged_step != self.simulation.step_count:
            self._write_metrics()

        if self._window is not None:
            self._window.close()

        self._file.close()
        self._closed = True

    def _process_events(self) -> None:
        """Process optional visualization events."""
        if self._window is None:
            return

        import pygame

        pressed = self._window.process_events()
        if pygame.K_s in pressed and self._renderer is not None:
            self._renderer.toggle_sensors()

        if not self._window.running:
            self.close()

    def _metrics_path(self, output_dir: Path) -> Path:
        path = Path(self.simulation.config.metrics.file)
        if path.is_absolute():
            return path
        return output_dir / path

    def _open_renderer(self) -> None:
        from evosim.renderers.pygame_common import PygameWindow
        from evosim.renderers.pygame_foraging import (
            DEFAULT_FPS,
            PygameForagingRenderer,
        )

        self._renderer = PygameForagingRenderer()
        self._window = PygameWindow(
            self.simulation.world_size,
            caption="EvoSim - Foraging",
            fps=DEFAULT_FPS,
            panel_width=self._renderer.panel_width,
        )

    def _write_metrics(self) -> None:
        metrics = self.simulation.metrics()
        self._writer.writerow(metrics)
        self._file.flush()
        self._last_logged_step = self.simulation.step_count

        print(
            f"step={metrics['step']} "
            f"population={metrics['population']} "
            f"food={metrics['food']} "
            f"births={metrics['births']} "
            f"deaths={metrics['deaths']} "
            f"mean_energy={metrics['mean_energy']:.1f}"
        )
