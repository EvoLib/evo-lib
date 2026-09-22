# SPDX-License-Identifier: MIT
"""World objects used by the built-in Predator-Prey simulation."""

from dataclasses import dataclass

from evolib import Indiv


@dataclass(slots=True, frozen=True)
class SensorRay:
    """One visual sensor ray relative to an agent heading."""

    angle: float
    range: float


@dataclass(slots=True)
class Agent:
    """Shared spatial and energy state for one evolvable agent."""

    indiv: Indiv
    x: float
    y: float
    heading: float
    energy: float
    radius: float


@dataclass(slots=True)
class Prey(Agent):
    """Prey that restores movement energy by grazing."""

    reproduction_timer: int = 0


@dataclass(slots=True)
class Predator(Agent):
    """Predator that survives and reproduces by capturing Prey."""

    feeding_cooldown: int = 0
    reproduction_cooldown: int = 0


@dataclass(slots=True, frozen=True)
class AgentAction:
    """Normalized movement command produced by an EvoNet controller."""

    turn: float
    throttle: float
