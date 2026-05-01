# SPDX-License-Identifier: MIT
"""Reusable simulation objects for the pixel-based LineFollower environment."""

import math
from dataclasses import dataclass

import pygame

from evolib.evolib_envs.core.sensors import SensorPoint, SensorPointState


@dataclass(frozen=True)
class LineSensor(SensorPoint):
    """LineFollower sensor with a circular pygame collision mask."""

    def build_mask(self) -> pygame.mask.Mask:
        """Build a circular mask for pixel-perfect line contact checks."""

        diameter = self.radius * 2 + 1
        surface = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
        pygame.draw.circle(
            surface,
            (255, 255, 255, 255),
            (self.radius, self.radius),
            self.radius,
        )
        return pygame.mask.from_surface(surface)


@dataclass
class LineFollowerRobot:
    """
    Robot body and sensor geometry for the LineFollower environment.

    This class owns robot movement and sensor contact logic.
    """

    x: float = 0.0
    y: float = 0.0
    angle: float = 0.0
    base_speed: float = 4.5
    turn_strength: float = 0.12
    radius: int = 10

    sensors: tuple[LineSensor, ...] = (
        LineSensor(forward=45.0, side=-20.0, radius=7),
        LineSensor(forward=45.0, side=20.0, radius=7),
    )

    def __post_init__(self) -> None:
        self.sensor_masks = [sensor.build_mask() for sensor in self.sensors]
        self.body_mask = self._build_body_mask()

    @property
    def sensor_radius(self) -> int:
        """Return the shared sensor radius used by the renderer."""

        return self.sensors[0].radius

    def reset(self, *, x: float, y: float, angle: float) -> None:
        """Place the robot at a new pose."""

        self.x = float(x)
        self.y = float(y)
        self.angle = float(angle)

    def step(self, turn: float) -> None:
        """Advance the robot using one clipped steering action."""

        clipped_turn = max(-1.0, min(1.0, float(turn)))

        self.angle += clipped_turn * self.turn_strength
        self.x += math.cos(self.angle) * self.base_speed
        self.y += math.sin(self.angle) * self.base_speed

    def _get_sensor_states(self, line_mask: pygame.mask.Mask) -> list[SensorPointState]:
        """Return all sensor positions and binary line-contact values."""

        states: list[SensorPointState] = []

        for sensor, mask in zip(self.sensors, self.sensor_masks):
            x, y = self.sensor_position(sensor)

            value = self.sensor_touches_line(
                sensor=sensor,
                sensor_mask=mask,
                line_mask=line_mask,
                x=x,
                y=y,
            )

            states.append(
                SensorPointState(
                    value=value,
                    x=x,
                    y=y,
                    radius=sensor.radius,
                )
            )

        return states

    def sensor_position(self, sensor: LineSensor) -> tuple[float, float]:
        """Return one sensor position in pixel coordinates."""

        forward_x = math.cos(self.angle)
        forward_y = math.sin(self.angle)

        right_x = -math.sin(self.angle)
        right_y = math.cos(self.angle)

        sensor_x = self.x + forward_x * sensor.forward + right_x * sensor.side
        sensor_y = self.y + forward_y * sensor.forward + right_y * sensor.side

        return sensor_x, sensor_y

    def touches_line(self, line_mask: pygame.mask.Mask) -> bool:
        """Return True if the robot body overlaps the line mask."""

        offset = (
            int(round(self.x)) - self.radius,
            int(round(self.y)) - self.radius,
        )
        return bool(line_mask.overlap(self.body_mask, offset))

    @staticmethod
    def sensor_touches_line(
        *,
        sensor: LineSensor,
        sensor_mask: pygame.mask.Mask,
        line_mask: pygame.mask.Mask,
        x: float,
        y: float,
    ) -> float:
        """Return 1.0 if the circular sensor overlaps the line mask, else 0.0."""

        offset = (
            int(round(x)) - sensor.radius,
            int(round(y)) - sensor.radius,
        )
        return 1.0 if line_mask.overlap(sensor_mask, offset) else 0.0

    def _build_body_mask(self) -> pygame.mask.Mask:
        """Create a circular mask representing the robot body."""

        diameter = self.radius * 2 + 1
        surface = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
        pygame.draw.circle(
            surface,
            (255, 255, 255, 255),
            (self.radius, self.radius),
            self.radius,
        )
        return pygame.mask.from_surface(surface)
