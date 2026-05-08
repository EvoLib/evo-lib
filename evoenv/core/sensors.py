# SPDX-License-Identifier: MIT
"""Shared sensor geometry helpers for evolib-envs."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SensorState:
    """Base sensor state for debugging and rendering."""

    value: float


@dataclass(frozen=True)
class SensorPointState(SensorState):
    """State of a point or circular area sensor."""

    x: float
    y: float
    radius: int


@dataclass(frozen=True)
class SensorLineState(SensorState):
    """State of a line-shaped sensor."""

    start_x: float
    start_y: float
    end_x: float
    end_y: float


@dataclass(frozen=True)
class SensorPoint:
    """Sensor defined relative to a moving body."""

    forward: float
    side: float
    radius: int


@dataclass(frozen=True)
class SensorLine:
    """Forward-facing line sensor."""

    range: float

    def get_state(
        self,
        *,
        x: float,
        y: float,
        value: float = 0.0,
    ) -> SensorLineState:
        """Return the current line sensor state."""

        return SensorLineState(
            start_x=x,
            start_y=y,
            end_x=x + self.range,
            end_y=y,
            value=value,
        )
