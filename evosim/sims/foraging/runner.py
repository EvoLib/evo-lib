# SPDX-License-Identifier: MIT
"""Runtime helpers for the built-in Foraging simulation."""

import csv
from pathlib import Path

from evosim.sims.foraging.config import ForagingConfig
from evosim.sims.foraging.simulation import ForagingSimulation


def _metrics_path(config: ForagingConfig, output_dir: Path) -> Path:
    path = Path(config.metrics.file)
    if path.is_absolute():
        return path
    return output_dir / path


def _write_metrics(
    writer: csv.DictWriter[str],
    simulation: ForagingSimulation,
) -> None:
    """Write one metrics row and print a compact status line."""
    metrics = simulation.metrics()
    writer.writerow(metrics)
    print(
        f"step={metrics['step']} "
        f"population={metrics['population']} "
        f"food={metrics['food']} "
        f"births={metrics['births']} "
        f"deaths={metrics['deaths']} "
        f"mean_energy={metrics['mean_energy']:.1f}"
    )


def _run_headless(
    simulation: ForagingSimulation,
    config: ForagingConfig,
    writer: csv.DictWriter[str],
) -> int:
    """Run the simulation without graphical output."""
    last_logged_step = simulation.step_count

    while simulation.step_count < config.steps and not simulation.is_extinct:
        simulation.step()
        if simulation.step_count % config.metrics.interval == 0:
            _write_metrics(writer, simulation)
            last_logged_step = simulation.step_count

    return last_logged_step


def _run_rendered(
    simulation: ForagingSimulation,
    config: ForagingConfig,
    writer: csv.DictWriter[str],
) -> int:
    """Run the simulation with Pygame visualization."""
    import pygame

    from evosim.renderers.pygame_common import PygameWindow
    from evosim.renderers.pygame_foraging import DEFAULT_FPS, PygameForagingRenderer

    renderer = PygameForagingRenderer()
    window = PygameWindow(
        simulation.world_size,
        caption="EvoSim - Foraging",
        fps=DEFAULT_FPS,
        panel_width=renderer.panel_width,
    )
    last_logged_step = simulation.step_count

    try:
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
            if simulation.step_count % config.metrics.interval == 0:
                _write_metrics(writer, simulation)
                last_logged_step = simulation.step_count

            renderer.draw(window.screen, simulation, window.font)
            window.update()
    finally:
        window.close()

    return last_logged_step


def run_foraging(
    config: ForagingConfig,
    *,
    render: bool = False,
    output_dir: str | Path = ".",
) -> None:
    """Run one configured Foraging simulation."""
    simulation = ForagingSimulation(config)
    metrics_path = _metrics_path(config, Path(output_dir))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(simulation.metrics()))
        writer.writeheader()
        _write_metrics(writer, simulation)

        if render:
            last_logged_step = _run_rendered(simulation, config, writer)
        else:
            last_logged_step = _run_headless(simulation, config, writer)

        if last_logged_step != simulation.step_count:
            _write_metrics(writer, simulation)
