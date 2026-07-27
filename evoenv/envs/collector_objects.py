# SPDX-License-Identifier: MIT
"""Reusable simulation objects for the Collector environment."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pygame


@dataclass
class CollectorAgent:
    """Circular agent with heading-based movement."""

    x: float
    y: float
    heading: float
    radius: int

    def move(
        self,
        *,
        turn: float,
        throttle: float,
        base_speed: float,
        turn_strength: float,
    ) -> None:
        """Move the agent one step using turn and throttle actions."""
        self.heading = (self.heading + turn * turn_strength) % (math.tau)
        speed = base_speed * throttle
        self.x += math.sin(self.heading) * speed
        self.y -= math.cos(self.heading) * speed

    @property
    def position(self) -> tuple[float, float]:
        """Return the current position as a tuple."""
        return (self.x, self.y)


@dataclass
class CollectorFood:
    """One collectible food item."""

    x: float
    y: float
    radius: int

    @property
    def position(self) -> tuple[float, float]:
        """Return the current position as a tuple."""
        return (self.x, self.y)


@dataclass
class CollectorObstacle:
    """Axis-aligned rectangular obstacle."""

    rect: pygame.Rect


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Return Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def circle_intersects_rect(
    *,
    x: float,
    y: float,
    radius: float,
    rect: pygame.Rect,
) -> bool:
    """Return True if a circle intersects an axis-aligned rectangle."""
    closest_x = max(float(rect.left), min(float(x), float(rect.right)))
    closest_y = max(float(rect.top), min(float(y), float(rect.bottom)))
    return math.hypot(x - closest_x, y - closest_y) <= radius
