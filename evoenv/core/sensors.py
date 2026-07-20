# SPDX-License-Identifier: MIT
"""Shared sensor geometry helpers for evoenv."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pygame
from collection.abc import Iterable
from evoenv.core.utils import clamp01


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


def ray_rect_hit_fraction(
    *,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    rect: pygame.Rect,
) -> float | None:
    """Return the first hit fraction of a ray against one rectangle."""
    ray_length = math.hypot(end_x - start_x, end_y - start_y)
    if ray_length <= 0.0:
        return None

    start = (int(round(start_x)), int(round(start_y)))
    end = (int(round(end_x)), int(round(end_y)))

    clipped_line = rect.clipline(start, end)
    if not clipped_line:
        return None

    hit_x, hit_y = clipped_line[0]
    hit_distance = math.hypot(float(hit_x) - start_x, float(hit_y) - start_y)

    return clamp01(hit_distance / ray_length)


def cast_ray_against_rects(
    sensor: RaySensor,
    pose: Pose2D,
    rects: Iterable[pygame.Rect],
) -> tuple[float, float | None]:
    """Return proximity value and nearest hit fraction for one ray sensor."""
    state = sensor.get_state(pose, value=0.0)
    first_hit_fraction: float | None = None

    for rect in rects:
        hit_fraction = ray_rect_hit_fraction(
            start_x=state.start_x,
            start_y=state.start_y,
            end_x=state.end_x,
            end_y=state.end_y,
            rect=rect,
        )
        if hit_fraction is None:
            continue

        if first_hit_fraction is None or hit_fraction < first_hit_fraction:
            first_hit_fraction = hit_fraction

    if first_hit_fraction is None:
        return 0.0, None

    return 1.0 - first_hit_fraction, first_hit_fraction
