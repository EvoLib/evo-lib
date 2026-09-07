# SPDX-License-Identifier: MIT
"""Base abstraction for persistent simulations."""

from abc import ABC, abstractmethod


class Simulation(ABC):
    """Base class for simulations that advance continuously in discrete steps."""

    def __init__(self) -> None:
        self.step_count = 0

    @abstractmethod
    def reset(self, *, seed: int | None = None) -> None:
        """Reset the complete simulation state."""

    @abstractmethod
    def step(self) -> None:
        """Advance the simulation by one synchronous step."""
