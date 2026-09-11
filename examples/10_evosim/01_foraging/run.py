# SPDX-License-Identifier: MIT
"""Run the built-in persistent Foraging simulation."""

import argparse
import csv
from pathlib import Path
from types import TracebackType
from typing import TextIO

from evosim.sims.foraging import ForagingConfig, ForagingSimulation

# CONFIG_PATH = Path(__file__).with_name("simulation.yaml")
CONFIG_PATH = "simulation.yaml"


class CsvMetricsLogger:
    """Write periodic simulation metrics to one CSV file."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO = path.open("w", encoding="utf-8", newline="")
        self._writer: csv.DictWriter[str] | None = None
        self.last_step: int | None = None

    def write(self, simulation: ForagingSimulation) -> None:
        """Write the current metrics, creating the CSV header on first use."""
        metrics = simulation.metrics()
        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=list(metrics))
            self._writer.writeheader()
        self._writer.writerow(metrics)
        self._file.flush()
        self.last_step = simulation.step_count

    def __enter__(self) -> "CsvMetricsLogger":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._file.close()


def _metrics_path(config: ForagingConfig) -> Path:
    path = Path(config.metrics.file)
    if path.is_absolute():
        return path
    return CONFIG_PATH.parent / path


def _log_if_due(
    logger: CsvMetricsLogger,
    simulation: ForagingSimulation,
    config: ForagingConfig,
) -> None:
    if simulation.step_count % config.metrics.interval == 0:
        _log_metrics(logger, simulation)


def _log_final(logger: CsvMetricsLogger, simulation: ForagingSimulation) -> None:
    if logger.last_step != simulation.step_count:
        _log_metrics(logger, simulation)


def _print_status(simulation: ForagingSimulation) -> None:
    """Print one compact simulation status line."""
    print(
        f"step={simulation.step_count} "
        f"population={simulation.population_size} "
        f"food={len(simulation.food)} "
        f"births={simulation.births} "
        f"deaths={simulation.deaths} "
        f"mean_energy={simulation.mean_energy:.1f}"
    )


def _log_metrics(logger: CsvMetricsLogger, simulation: ForagingSimulation) -> None:
    logger.write(simulation)
    _print_status(simulation)


def run_headless(config: ForagingConfig) -> None:
    """Run the configured simulation without creating a graphical window."""
    simulation = ForagingSimulation(config)

    with CsvMetricsLogger(_metrics_path(config)) as logger:
        _log_metrics(logger, simulation)
        while simulation.step_count < config.steps and not simulation.is_extinct:
            simulation.step()
            _log_if_due(logger, simulation, config)
        _log_final(logger, simulation)


def run_rendered(config: ForagingConfig) -> None:
    """Run the configured simulation with optional Pygame visualization."""
    import pygame

    from evosim.renderers.pygame_common import PygameWindow
    from evosim.renderers.pygame_foraging import DEFAULT_FPS, PygameForagingRenderer

    simulation = ForagingSimulation(config)
    renderer = PygameForagingRenderer()
    window = PygameWindow(
        simulation.world_size,
        caption="EvoSim - Foraging",
        fps=DEFAULT_FPS,
        panel_width=renderer.panel_width,
    )

    try:
        with CsvMetricsLogger(_metrics_path(config)) as logger:
            _log_metrics(logger, simulation)
            while (
                window.running
                and simulation.step_count < config.steps
                and not simulation.is_extinct
            ):
                pressed = window.process_events()
                if pygame.K_s in pressed:
                    renderer.toggle_sensors()
                if not window.running:
                    break

                simulation.step()
                _log_if_due(logger, simulation, config)
                renderer.draw(window.screen, simulation, window.font)
                window.update()

            _log_final(logger, simulation)
    finally:
        window.close()


def parse_args() -> argparse.Namespace:
    """Parse visualization-only command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--render",
        action="store_true",
        help="show the Pygame visualization; headless execution is the default",
    )
    return parser.parse_args()


def main() -> None:
    """Run Foraging from its YAML configuration."""
    args = parse_args()
    config = ForagingConfig.from_yaml(CONFIG_PATH)

    if args.render:
        run_rendered(config)
    else:
        run_headless(config)


if __name__ == "__main__":
    main()
