# SPDX-License-Identifier: MIT
"""Shared sensor geometry helpers for evoenv."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .utils import clamp01


@dataclass(frozen=True)
class Pose2D:
    """
    Position and heading of an object in 2D space.

    The heading is stored in radians. A heading of 0 points upward.
    """

    x: float
    y: float
    heading: float = 0.0


@dataclass(frozen=True)
class SensorState:
    """Base sensor state for debugging and rendering."""

    value: float


@dataclass(frozen=True)
class SensorPointState(SensorState):
    """State of a point-shaped sensor in world coordinates."""

    x: float
    y: float
    radius: int


@dataclass(frozen=True)
class SensorLineState(SensorState):
    """State of a line-shaped sensor in world coordinates."""

    start_x: float
    start_y: float
    end_x: float
    end_y: float


@dataclass(frozen=True)
class PointSensor:
    """
    A circular point sensor defined relative to an object pose.

    Args:
        forward: Distance along the object's forward direction.
        side: Distance along the object's right-hand direction.
        radius: Sensor radius in world units.
    """

    forward: float
    side: float
    radius: int

    def get_state(
        self,
        pose: Pose2D,
        *,
        value: float = 0.0,
    ) -> SensorPointState:
        """Return the visible sensor point in world coordinates."""
        forward_x = math.sin(pose.heading)
        forward_y = -math.cos(pose.heading)

        right_x = math.cos(pose.heading)
        right_y = math.sin(pose.heading)

        return SensorPointState(
            value=value,
            x=pose.x + forward_x * self.forward + right_x * self.side,
            y=pose.y + forward_y * self.forward + right_y * self.side,
            radius=int(self.radius),
        )


@dataclass(frozen=True)
class RaySensor:
    """
    A ray sensor defined relative to an object pose.

    Args:
        length: Maximum sensor length in world units.
        angle: Relative angle in radians.
    """

    length: float
    angle: float

    def get_state(
        self,
        pose: Pose2D,
        *,
        value: float = 0.0,
        hit_fraction: float | None = None,
    ) -> SensorLineState:
        """Return the visible sensor ray in world coordinates."""
        visible_length = self.length

        if hit_fraction is not None:
            visible_length *= clamp01(hit_fraction)

        absolute_angle = pose.heading + self.angle

        dx = math.sin(absolute_angle) * visible_length
        dy = -math.cos(absolute_angle) * visible_length

        return SensorLineState(
            value=value,
            start_x=pose.x,
            start_y=pose.y,
            end_x=pose.x + dx,
            end_y=pose.y + dy,
        )
