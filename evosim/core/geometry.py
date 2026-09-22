# SPDX-License-Identifier: MIT
"""Geometry helpers for toroidal EvoSim worlds."""

import math


def wrap_coordinate(value: float, size: float) -> float:
    """Wrap one coordinate into the half-open interval ``[0, size)``."""
    if size <= 0.0:
        raise ValueError("size must be greater than zero.")
    return value % size


def toroidal_delta(source: float, target: float, size: float) -> float:
    """Return the shortest signed displacement from source to target."""
    if size <= 0.0:
        raise ValueError("size must be greater than zero.")
    return (target - source + size / 2.0) % size - size / 2.0


def toroidal_displacement(
    source_x: float,
    source_y: float,
    target_x: float,
    target_y: float,
    width: float,
    height: float,
) -> tuple[float, float]:
    """Return the shortest 2D displacement in a toroidal world."""
    return (
        toroidal_delta(source_x, target_x, width),
        toroidal_delta(source_y, target_y, height),
    )


def toroidal_distance_squared(
    source_x: float,
    source_y: float,
    target_x: float,
    target_y: float,
    width: float,
    height: float,
) -> float:
    """Return squared toroidal distance between two points."""
    dx, dy = toroidal_displacement(
        source_x,
        source_y,
        target_x,
        target_y,
        width,
        height,
    )
    return dx * dx + dy * dy


def angle_delta(source: float, target: float) -> float:
    """Return the shortest signed angular displacement from source to target."""
    return (target - source + math.pi) % math.tau - math.pi
