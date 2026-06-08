# SPDX-License-Identifier: MIT
"""Reusable simulation objects for the pixel-based LineFollower environment."""

import math
from dataclasses import dataclass, field

import pygame
from evoenv.core.sensors import PointSensor, Pose2D, SensorPointState
from evoenv.core.utils import clamp


@dataclass(frozen=True)
class LineSensor(PointSensor):
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


DEFAULT_LINE_SENSORS: tuple[LineSensor, ...] = (
    LineSensor(forward=45.0, side=-20.0, radius=7),
    LineSensor(forward=45.0, side=20.0, radius=7),
)


@dataclass
class LineFollowerRobot:
    """
    Robot body and sensor geometry for the LineFollower environment.

    This class owns robot movement, body contact checks, and sensor contact checks.
    """

    x: float = 0.0
    y: float = 0.0
    angle: float = 0.0
    base_speed: float = 4.5
    turn_strength: float = 0.12
    radius: int = 10

    sensors: tuple[LineSensor, ...] = DEFAULT_LINE_SENSORS

    sensor_masks: tuple[pygame.mask.Mask, ...] = field(init=False, repr=False)
    body_mask: pygame.mask.Mask = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.sensors:
            raise ValueError("LineFollowerRobot requires at least one sensor.")

        self.sensor_masks = tuple(sensor.build_mask() for sensor in self.sensors)
        self.body_mask = self._build_body_mask()

    @property
    def pose(self) -> Pose2D:
        """Return the robot pose using evoenv's upward-facing heading convention."""
        return Pose2D(
            x=self.x,
            y=self.y,
            heading=self.angle + math.pi / 2.0,
        )

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
        clamped_turn = clamp(turn, -1.0, 1.0)

        self.angle += clamped_turn * self.turn_strength
        self.x += math.cos(self.angle) * self.base_speed
        self.y += math.sin(self.angle) * self.base_speed

    def get_sensor_states(self, line_mask: pygame.mask.Mask) -> list[SensorPointState]:
        """Return all sensor positions and binary line-contact values."""
        states: list[SensorPointState] = []

        for sensor, sensor_mask in zip(self.sensors, self.sensor_masks):
            state = sensor.get_state(self.pose)

            value = self.sensor_touches_line(
                sensor_mask=sensor_mask,
                line_mask=line_mask,
                x=state.x,
                y=state.y,
                radius=state.radius,
            )

            states.append(sensor.get_state(self.pose, value=value))

        return states

    def sensor_position(self, sensor: LineSensor) -> tuple[float, float]:
        """Return one sensor position in pixel coordinates."""
        state = sensor.get_state(self.pose)
        return state.x, state.y

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
        sensor_mask: pygame.mask.Mask,
        line_mask: pygame.mask.Mask,
        x: float,
        y: float,
        radius: int,
    ) -> float:
        """Return 1.0 if the circular sensor overlaps the line mask, else 0.0."""
        offset = (
            int(round(x)) - int(radius),
            int(round(y)) - int(radius),
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
