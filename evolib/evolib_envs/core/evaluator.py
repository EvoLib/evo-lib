# SPDX-License-Identifier: MIT
from evolib.evolib_envs.core.controller import Controller
from evolib.evolib_envs.core.env import Env, validate_action, validate_observation


def evaluate_episode(
    env: Env,
    controller: Controller,
    *,
    seed: int | None = None,
    max_steps: int = 500,
) -> float:
    """Run one episode and return accumulated reward (fitness)."""

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
