# SPDX-License-Identifier: MIT
"""
Core environment protocol for evolib-envs.

This module defines minimal interfaces for single-agent and batched environments
(TODO: MultiAgentEnv)

The interface follows the common reinforcement-learning style:

    observation = env.reset(seed=42)
    observation, reward, done, info = env.step(action)
"""

from typing import Any, Protocol, TypeAlias, runtime_checkable

InfoDict: TypeAlias = dict[str, Any]

Observation: TypeAlias = list[float]
Action: TypeAlias = list[float]
StepResult: TypeAlias = tuple[Observation, float, bool, InfoDict]

BatchObservation: TypeAlias = list[Observation]
BatchAction: TypeAlias = list[Action]
BatchReward: TypeAlias = list[float]
BatchDone: TypeAlias = list[bool]
BatchInfo: TypeAlias = list[InfoDict]
BatchStepResult: TypeAlias = tuple[
    BatchObservation,
    BatchReward,
    BatchDone,
    BatchInfo,
]


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


@runtime_checkable
class BatchEnv(Protocol):
    """
    Protocol for batched single-agent environments.

    A BatchEnv runs multiple independent episodes in parallel. This is useful for
    evolutionary evaluation, where many individuals are evaluated under the same task
    structure but do not interact with each other.

    This is not a multi-agent environment. Agents in a BatchEnv do not share world state
    unless a concrete implementation explicitly documents otherwise.
    """

    observation_size: int
    action_size: int
    batch_size: int

    def reset(self, seed: int | None = None) -> BatchObservation:
        """Reset all episodes and return one observation per batch item."""
        ...

    def step(self, actions: BatchAction) -> BatchStepResult:
        """
        Advance all episodes by one step.

        Args:
            actions: One action per batch item. Length must match ``batch_size``.

        Returns:
            A tuple ``(observations, rewards, dones, infos)``.
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


def validate_batch_observation(
    env: BatchEnv,
    observations: BatchObservation,
) -> None:
    """Validate batched observations against ``env.batch_size`` and
    ``env.observation_size``."""
    if not isinstance(observations, list):
        raise TypeError("Batch observation must be a list of observations.")

    if len(observations) != env.batch_size:
        raise ValueError(
            "Invalid batch observation size: "
            f"expected {env.batch_size}, got {len(observations)}."
        )

    for index, observation in enumerate(observations):
        if not isinstance(observation, list):
            raise TypeError(f"Observation at index {index} must be a list[float].")

        if len(observation) != env.observation_size:
            raise ValueError(
                "Invalid observation size at index "
                f"{index}: expected {env.observation_size}, "
                f"got {len(observation)}."
            )


def validate_batch_action(env: BatchEnv, actions: BatchAction) -> None:
    """Validate batched actions against ``env.batch_size`` and ``env.action_size``."""
    if not isinstance(actions, list):
        raise TypeError("Batch action must be a list of actions.")

    if len(actions) != env.batch_size:
        raise ValueError(
            "Invalid batch action size: "
            f"expected {env.batch_size}, got {len(actions)}."
        )

    for index, action in enumerate(actions):
        if not isinstance(action, list):
            raise TypeError(f"Action at index {index} must be a list[float].")

        if len(action) != env.action_size:
            raise ValueError(
                "Invalid action size at index "
                f"{index}: expected {env.action_size}, got {len(action)}."
            )
