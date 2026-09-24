# SPDX-License-Identifier: MIT
"""Pygame renderer for the built-in Predator-Prey simulation."""

import math

import pygame

from evosim.renderers.pygame_common import (
    DEFAULT_INFO_PANEL_WIDTH,
    draw_text_panel,
    split_simulation_screen,
)
from evosim.sims.predator_prey.objects import Agent
from evosim.sims.predator_prey.simulation import PredatorPreySimulation

_PREY_LOW_COLOR = (31, 58, 95)
_PREY_HIGH_COLOR = (86, 204, 242)
_PREDATOR_LOW_COLOR = (110, 42, 42)
_PREDATOR_HIGH_COLOR = (242, 145, 70)
_RAY_COLOR = (120, 120, 120)


class PygamePredatorPreyRenderer:
    """Render Predator-Prey state with optional ray-sensor overlays."""

    def __init__(self, panel_width: int = DEFAULT_INFO_PANEL_WIDTH) -> None:
        self.panel_width = panel_width
        self.show_sensors = False

    def toggle_sensors(self) -> None:
        """Toggle visualization of agent sensor rays."""
        self.show_sensors = not self.show_sensors

    def draw(
        self,
        screen: pygame.Surface,
        simulation: PredatorPreySimulation,
        font: pygame.font.Font,
        *,
        title: str = "EvoSim Predator-Prey",
    ) -> None:
        """Draw one complete Predator-Prey frame."""
        world_screen, info_screen = split_simulation_screen(
            screen,
            simulation.world_size,
            self.panel_width,
        )
        self._draw_world(world_screen, simulation)
        self._draw_info(info_screen, simulation, font, title)

    def _draw_world(
        self,
        screen: pygame.Surface,
        simulation: PredatorPreySimulation,
    ) -> None:
        screen.fill((20, 20, 20))

        if self.show_sensors:
            for prey in simulation.prey:
                self._draw_rays(screen, simulation, prey)
            for predator in simulation.predators:
                self._draw_rays(screen, simulation, predator)

        for prey in simulation.prey:
            self._draw_agent(
                screen,
                prey,
                simulation.config.prey.energy_capacity,
                _PREY_LOW_COLOR,
                _PREY_HIGH_COLOR,
                (225, 240, 255),
            )

        for predator in simulation.predators:
            self._draw_agent(
                screen,
                predator,
                simulation.config.predator.energy_capacity,
                _PREDATOR_LOW_COLOR,
                _PREDATOR_HIGH_COLOR,
                (255, 235, 210),
            )

    def _draw_rays(
        self,
        screen: pygame.Surface,
        simulation: PredatorPreySimulation,
        agent: Agent,
    ) -> None:
        width, height = simulation.world_size

        for ray in simulation.ray_layout(agent):
            self._draw_toroidal_ray(
                screen,
                agent,
                agent.heading + ray.angle,
                ray.range,
                width,
                height,
            )

    @staticmethod
    def _draw_toroidal_ray(
        screen: pygame.Surface,
        agent: Agent,
        angle: float,
        length: float,
        width: int,
        height: int,
    ) -> None:
        dx = math.cos(angle) * length
        dy = math.sin(angle) * length

        for offset_x in (-width, 0, width):
            for offset_y in (-height, 0, height):
                start = (
                    int(round(agent.x + offset_x)),
                    int(round(agent.y + offset_y)),
                )
                end = (
                    int(round(agent.x + offset_x + dx)),
                    int(round(agent.y + offset_y + dy)),
                )
                pygame.draw.line(screen, _RAY_COLOR, start, end, 1)

    @staticmethod
    def _draw_agent(
        screen: pygame.Surface,
        agent: Agent,
        energy_capacity: float,
        low_color: tuple[int, int, int],
        high_color: tuple[int, int, int],
        heading_color: tuple[int, int, int],
    ) -> None:
        energy_ratio = min(max(agent.energy / energy_capacity, 0.0), 1.0)
        color = tuple(
            int(round(low + (high - low) * energy_ratio))
            for low, high in zip(low_color, high_color, strict=True)
        )

        position = (int(round(agent.x)), int(round(agent.y)))
        radius = int(round(agent.radius))
        pygame.draw.circle(screen, color, position, radius)

        heading_length = agent.radius * 1.8
        heading_end = (
            int(round(agent.x + math.cos(agent.heading) * heading_length)),
            int(round(agent.y + math.sin(agent.heading) * heading_length)),
        )
        pygame.draw.line(screen, heading_color, position, heading_end, 2)

    def _draw_info(
        self,
        screen: pygame.Surface,
        simulation: PredatorPreySimulation,
        font: pygame.font.Font,
        title: str,
    ) -> None:
        prey_sensor = simulation.config.modules.prey
        predator_sensor = simulation.config.modules.predator
        lines = [
            title,
            f"step={simulation.step_count}",
            f"prey={simulation.prey_population_size}",
            f"predators={simulation.predator_population_size}",
            f"prey_births={simulation.prey_births}",
            f"predator_births={simulation.predator_births}",
            f"predator_deaths={simulation.predator_deaths}",
            f"captured={simulation.prey_captured}",
            f"prey_energy={simulation.mean_prey_energy:.1f}",
            f"predator_energy={simulation.mean_predator_energy:.1f}",
            "",
            (
                f"prey_sensor={math.degrees(prey_sensor.sensor_fov):.0f}deg / "
                f"{prey_sensor.sensor_range:.0f}"
            ),
            (
                f"pred_sensor={math.degrees(predator_sensor.sensor_fov):.0f}deg / "
                f"{predator_sensor.sensor_range:.0f}"
            ),
            "",
            f"sensors={'on' if self.show_sensors else 'off'}",
            "S: toggle sensors",
            "ESC: quit",
        ]
        draw_text_panel(screen, font, lines)
