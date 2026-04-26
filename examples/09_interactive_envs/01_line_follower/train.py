# SPDX-License-Identifier: MIT
"""Train an EvoLib population on LineFollowerEnv."""

from evolib import Indiv, Pop, save_best_indiv
from evolib.evolib_envs.core.env import Action, Observation
from evolib.evolib_envs.core.evaluator import evaluate_episode
from evolib.evolib_envs.envs.line_follower import LineFollowerEnv

CONFIG_FILE = "config.yaml"
MAX_STEPS = 1400
SEED = 42


class LineFollowerController:
    """Interpret an EvoLib individual as a line-follower controller."""

    def __init__(self, indiv: Indiv, brain_module: str = "brain") -> None:
        self.net = indiv.para[brain_module]

    def act(self, observation: Observation) -> Action:
        return self.net.calc(observation)


def eval_line_follower_fitness(indiv: Indiv) -> None:
    """Evaluate one individual on one LineFollower episode."""

    env = LineFollowerEnv()
    controller = LineFollowerController(indiv)

    total_reward = evaluate_episode(
        env,
        controller,
        seed=SEED,
        max_steps=MAX_STEPS,
    )

    indiv.fitness = -total_reward


def main() -> None:
    pop = Pop(config_path=CONFIG_FILE, fitness_function=eval_line_follower_fitness)
    pop.run(verbosity=1)
    save_best_indiv(pop, run_name="line_follower")


if __name__ == "__main__":
    main()
