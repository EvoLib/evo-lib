# SPDX-License-Identifier: MIT
from collections.abc import Callable
from typing import Protocol

from evoenv.core.env import Action, Observation

RuleFunction = Callable[[Observation], Action]


class Controller(Protocol):
    """Maps observations to actions."""

    def act(self, observation: Observation) -> Action:
        """Return an action for the given observation."""
        ...


class CallbackController:
    """Wrap a rule function as a controller."""

    def __init__(self, rule: RuleFunction) -> None:
        self.rule = rule

    def act(self, observation: Observation) -> Action:
        """Return the action produced by the wrapped rule."""

        return self.rule(observation)
