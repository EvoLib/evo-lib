# SPDX-License-Identifier: MIT
"""Evaluation helpers for episodic environments."""

from typing import Callable, Generic, Protocol, TypeVar

from evoenv.core.controller import Controller
from evoenv.core.env import Env, validate_action, validate_observation


class FitnessAssignable(Protocol):
    """Minimal protocol for objects that can receive a fitness value."""

    fitness: float | None


IndividualT = TypeVar("IndividualT", bound=FitnessAssignable)
EnvFactory = Callable[[], Env]
ControllerFactory = Callable[[IndividualT], Controller]
RewardToFitness = Callable[[float], float]


def evaluate_episode(
    env: Env,
    controller: Controller,
    *,
    seed: int | None = None,
    max_steps: int = 500,
) -> float:
    """Run one episode and return accumulated reward."""

    observation = env.reset(seed=seed)
    validate_observation(env, observation)

    total_reward = 0.0

    for _ in range(max_steps):
        action = controller.act(observation)
        validate_action(env, action)

        observation, reward, done, _ = env.step(action)
        validate_observation(env, observation)

        total_reward += reward

        if done:
            break

    return total_reward


class EpisodeEvaluator(Generic[IndividualT]):
    """
    Generic fitness evaluator for episodic environments.

    The evaluator creates a fresh environment and controller for every individual, runs
    one episode, converts the accumulated reward into a fitness value, and writes that
    value back to the individual.
    """

    def __init__(
        self,
        *,
        env_factory: EnvFactory,
        controller_factory: ControllerFactory[IndividualT],
        seed: int | None = None,
        max_steps: int = 500,
        reward_to_fitness: RewardToFitness | None = None,
    ) -> None:
        self.env_factory = env_factory
        self.controller_factory = controller_factory
        self.seed = seed
        self.max_steps = max_steps
        self.reward_to_fitness = reward_to_fitness or self._minimize_negative_reward

    def __call__(self, individual: IndividualT) -> None:
        """Evaluate one individual and assign its fitness value."""

        env = self.env_factory()
        controller = self.controller_factory(individual)

        total_reward = evaluate_episode(
            env,
            controller,
            seed=self.seed,
            max_steps=self.max_steps,
        )

        individual.fitness = self.reward_to_fitness(total_reward)

    @staticmethod
    def _minimize_negative_reward(total_reward: float) -> float:
        """Convert reward maximization into EvoLib's minimization convention."""

        return -total_reward
