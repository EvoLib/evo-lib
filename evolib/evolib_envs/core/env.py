# SPDX-License-Identifier: MIT
"""
Core environment protocol for evolib-envs.

This module defines the interface used by interactive and headless
environments.

The interface follows the common reinforcement-learning style:

    observation = env.reset(seed=42)
    observation, reward, done, info = env.step(action)

Concrete environments should keep simulation logic here and place rendering,
keyboard input, and framework-specific adapters in separate modules.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

InfoDict = dict[str, Any]
Observation = list[float]
Action = list[float]
StepResult = tuple[Observation, float, bool, InfoDict]


@runtime_checkable
class Env(Protocol):
    """
    Minimal protocol for an evolib-envs environment.

    Implementations represent the task logic only:
    - simulation state
    - observations
    - actions
    - reward calculation
    - termination condition
    """

    observation_size: int
    action_size: int

    def reset(self, seed: int | None = None) -> Observation:
        """
        Reset the environment and return the initial observation.

        Args:
            seed: Optional random seed for reproducible episodes.

        Returns:
            The initial observation as a flat list of floats.
        """
        ...

    def step(self, action: Action) -> StepResult:
        """
        Advance the environment by one step.

        Args:
            action: Flat list of action values. The expected length must match
                ``action_size``.

        Returns:
            A tuple ``(observation, reward, done, info)``:
            - observation: next observation as a flat list of floats
            - reward: scalar reward for this step
            - done: True if the episode has ended
            - info: optional diagnostic values for debugging, rendering, or logging
        """
        ...


def validate_observation(env: Env, observation: Observation) -> None:
    """
    Validate an observation against ``env.observation_size``.

    Args:
        env: Environment instance.
        observation: Observation returned by the environment.

    Raises:
        TypeError: If the observation is not a list.
        ValueError: If the observation length is invalid.
    """
    if not isinstance(observation, list):
        raise TypeError("Observation must be a list[float].")

    if len(observation) != env.observation_size:
        raise ValueError(
            "Invalid observation size: "
            f"expected {env.observation_size}, got {len(observation)}."
        )


def validate_action(env: Env, action: Action) -> None:
    """
    Validate an action against ``env.action_size``.

    Args:
        env: Environment instance.
        action: Action passed to the environment.

    Raises:
        TypeError: If the action is not a list.
        ValueError: If the action length is invalid.
    """
    if not isinstance(action, list):
        raise TypeError("Action must be a list[float].")

    if len(action) != env.action_size:
        raise ValueError(
            "Invalid action size: " f"expected {env.action_size}, got {len(action)}."
        )
