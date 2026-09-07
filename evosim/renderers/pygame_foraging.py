# SPDX-License-Identifier: MIT
"""Pygame renderer for the built-in Foraging simulation."""

import math

import pygame

from evosim.renderers.pygame_common import (
    DEFAULT_INFO_PANEL_WIDTH,
    draw_text_panel,
    split_simulation_screen,
)
from evosim.sims.foraging import Forager, ForagingSimulation

DEFAULT_FPS = 60
_LOW_ENERGY_COLOR = (31, 58, 95)
_HIGH_ENERGY_COLOR = (86, 204, 242)
_SENSOR_COLORS = (
    (80, 180, 233),
    (230, 159, 0),
    (204, 121, 167),
)


class PygameForagingRenderer:
    """Render Foraging state with an optional local-sensor overlay."""

    def __init__(self, panel_width: int = DEFAULT_INFO_PANEL_WIDTH) -> None:
        self.panel_width = panel_width
        self.show_sensors = False

    def toggle_sensors(self) -> None:
        """Toggle visualization of Forager sensor fields."""
        self.show_sensors = not self.show_sensors

    def draw(
        self,
        screen: pygame.Surface,
        simulation: ForagingSimulation,
        font: pygame.font.Font,
        *,
        title: str = "EvoSim Foraging",
    ) -> None:
        """Draw one complete Foraging frame."""
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
        simulation: ForagingSimulation,
    ) -> None:
        screen.fill((20, 20, 20))

        if self.show_sensors:
            for forager in simulation.foragers:
                self._draw_sensors(screen, simulation, forager)

        for resource in simulation.food:
            pygame.draw.circle(
                screen,
                (80, 220, 120),
                (int(round(resource.x)), int(round(resource.y))),
                int(round(resource.radius)),
            )

        for forager in simulation.foragers:
            self._draw_forager(screen, simulation, forager)

    def _draw_sensors(
        self,
        screen: pygame.Surface,
        simulation: ForagingSimulation,
        forager: Forager,
    ) -> None:
        width, height = simulation.world_size

        for index, sensor in enumerate(simulation.sensor_layout(forager)):
            color = _SENSOR_COLORS[index % len(_SENSOR_COLORS)]
            center_color = (color[0] // 2, color[1] // 2, color[2] // 2)
            center = forager.heading + sensor.angle
            half_fov = sensor.fov / 2.0

            if sensor.fov < math.tau:
                for edge in (center - half_fov, center + half_fov):
                    self._draw_toroidal_ray(
                        screen,
                        forager,
                        edge,
                        sensor.range,
                        width,
                        height,
                        color,
                    )

                self._draw_toroidal_ray(
                    screen,
                    forager,
                    center,
                    sensor.range,
                    width,
                    height,
                    center_color,
                )

            self._draw_toroidal_arc(
                screen,
                forager,
                center,
                sensor.fov,
                sensor.range,
                width,
                height,
                color,
            )

    @staticmethod
    def _draw_toroidal_arc(
        screen: pygame.Surface,
        forager: Forager,
        angle: float,
        fov: float,
        radius: float,
        width: int,
        height: int,
        color: tuple[int, int, int],
    ) -> None:
        """Close a sensor field with an arc and its wrapped copies."""
        diameter = int(round(2.0 * radius))
        # Pygame arcs turn opposite to the screen-space sensor angles.
        start_angle = -(angle + fov / 2.0)
        stop_angle = -(angle - fov / 2.0)

        for offset_x in (-width, 0, width):
            for offset_y in (-height, 0, height):
                bounds = pygame.Rect(
                    int(round(forager.x + offset_x - radius)),
                    int(round(forager.y + offset_y - radius)),
                    diameter,
                    diameter,
                )
                pygame.draw.arc(screen, color, bounds, start_angle, stop_angle, 1)

    @staticmethod
    def _draw_toroidal_ray(
        screen: pygame.Surface,
        forager: Forager,
        angle: float,
        length: float,
        width: int,
        height: int,
        color: tuple[int, int, int],
    ) -> None:
        """Draw a sensor ray and its wrapped copies at world boundaries."""
        dx = math.cos(angle) * length
        dy = math.sin(angle) * length

        for offset_x in (-width, 0, width):
            for offset_y in (-height, 0, height):
                start = (
                    int(round(forager.x + offset_x)),
                    int(round(forager.y + offset_y)),
                )
                end = (
                    int(round(forager.x + offset_x + dx)),
                    int(round(forager.y + offset_y + dy)),
                )
                pygame.draw.line(screen, color, start, end, 1)

    def _draw_forager(
        self,
        screen: pygame.Surface,
        simulation: ForagingSimulation,
        forager: Forager,
    ) -> None:
        energy_ratio = min(
            max(
                forager.energy / simulation.config.forager.reproduction_threshold,
                0.0,
            ),
            1.0,
        )
        color = tuple(
            int(round(low + (high - low) * energy_ratio))
            for low, high in zip(_LOW_ENERGY_COLOR, _HIGH_ENERGY_COLOR, strict=True)
        )

        position = (int(round(forager.x)), int(round(forager.y)))
        radius = int(round(forager.radius))
        pygame.draw.circle(screen, color, position, radius)

        heading_length = forager.radius * 1.8
        heading_end = (
            int(round(forager.x + math.cos(forager.heading) * heading_length)),
            int(round(forager.y + math.sin(forager.heading) * heading_length)),
        )
        pygame.draw.line(screen, (225, 240, 255), position, heading_end, 2)

    def _draw_info(
        self,
        screen: pygame.Surface,
        simulation: ForagingSimulation,
        font: pygame.font.Font,
        title: str,
    ) -> None:
        lines = [
            title,
            f"step={simulation.step_count}",
            f"population={simulation.population_size}",
            f"food={len(simulation.food)}",
            f"births={simulation.births}",
            f"deaths={simulation.deaths}",
            f"food_eaten={simulation.food_eaten}",
            f"mean_energy={simulation.mean_energy:.1f}",
            f"oldest={simulation.oldest_lifetime_steps}",
            "",
            f"sensors={'on' if self.show_sensors else 'off'}",
            "S: toggle sensors",
            "ESC: quit",
        ]
        draw_text_panel(screen, font, lines)
