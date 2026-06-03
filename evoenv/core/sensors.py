# SPDX-License-Identifier: MIT
"""Shared sensor geometry helpers for evoenv."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Pose2D:
    """
    Position and heading of an object in 2D space.

    The heading is stored in radians. A heading of 0 points upward by default.
    """

    x: float
    y: float
    heading: float = 0.0


@dataclass(frozen=True)
class SensorState:
    """Base sensor state for debugging and rendering."""

    value: float


@dataclass(frozen=True)
class SensorLineState(SensorState):
    """State of a line-shaped sensor in world coordinates."""

    start_x: float
    start_y: float
    end_x: float
    end_y: float


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
            visible_length *= max(0.0, min(1.0, float(hit_fraction)))

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
