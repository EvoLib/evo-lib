# SPDX-License-Identifier: MIT
"""World objects used by the built-in Foraging simulation."""

from dataclasses import dataclass

from evolib import Indiv


@dataclass(slots=True)
class Food:
    """One consumable resource item."""

    x: float
    y: float
    radius: float


@dataclass(slots=True, frozen=True)
class FoodSensor:
    """Decoded local food sensor carried by one Forager."""

    angle: float
    fov: float
    range: float


@dataclass(slots=True)
class Forager:
    """World embodiment of one evolvable EvoLib individual."""

    indiv: Indiv
    x: float
    y: float
    heading: float
    energy: float
    radius: float
    lifetime_steps: int = 0


@dataclass(slots=True, frozen=True)
class ForagerAction:
    """Normalized movement command produced by a Forager controller."""

    turn: float
    throttle: float
