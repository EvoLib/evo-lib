# SPDX-License-Identifier: MIT
from typing import Protocol

from evolib.evolib_envs.core.env import Action, Observation


class Controller(Protocol):
    """Maps observations to actions."""

    def act(self, observation: Observation) -> Action:
        """Return an action for the given observation."""
        ...
