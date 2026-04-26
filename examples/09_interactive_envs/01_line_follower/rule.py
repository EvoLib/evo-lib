# SPDX-License-Identifier: MIT
"""Rule-based controller."""

from evolib.evolib_envs.core.env import Action, Observation
from evolib.evolib_envs.core.evaluator import evaluate_episode
from evolib.evolib_envs.envs.line_follower import LineFollowerEnv


class RuleBasedLineFollowerController:
    """Simple baseline controller for the line follower."""

    def act(self, observation: Observation) -> Action:
        left_sensor, right_sensor = observation

        error = right_sensor - left_sensor

        base = 0.6
        turn = 0.4

        left = base - error * turn
        right = base + error * turn

        return [left, right]


def main() -> None:
    env = LineFollowerEnv()
    controller = RuleBasedLineFollowerController()

    fitness = evaluate_episode(env, controller, seed=42, max_steps=500)

    print(f"Rule-based fitness: {fitness:.3f}")


if __name__ == "__main__":
    main()
