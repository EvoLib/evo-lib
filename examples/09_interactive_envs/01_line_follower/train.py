# SPDX-License-Identifier: MIT
"""Train an EvoLib population on LineFollowerEnv."""

import argparse
from typing import Callable

from render import run_debug_episode

from evolib import Indiv, Pop, save_best_indiv
from evolib.evolib_envs.cli import add_debug_arg
from evolib.evolib_envs.core.env import Action, Observation
from evolib.evolib_envs.core.evaluator import evaluate_episode
from evolib.evolib_envs.envs.line_follower import LineFollowerEnv

RUN_NAME = "line_follower"

CONFIG_FILE = "config.yaml"
MAX_STEPS = 500
SEED = 42

DEBUG_EVERY_N_GENERATIONS = 5
DEBUG_MAX_STEPS = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_debug_arg(parser)
    return parser.parse_args()


class LineFollowerController:
    """Interpret EvoNet output as steering."""

    def __init__(self, indiv: Indiv):
        self.net = indiv.para["brain"]

    def act(self, observation: Observation) -> Action:
        output = self.net.calc(observation)
        turn = float(output[0])
        turn = max(-1.0, min(1.0, turn))
        return [turn]


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


def make_debug_hook(debug_enabled: bool) -> Callable:
    """Create a generation-end hook for optional debug rendering."""

    def on_generation_end(pop: Pop) -> None:
        best = pop.best(sort=True)

        run_debug_episode(
            LineFollowerEnv(),
            LineFollowerController(best),
            enabled=debug_enabled,
            generation=pop.generation_num,
            title=f"Training Debug - Gen {pop.generation_num}",
        )

    return on_generation_end


def main() -> None:

    args = parse_args()

    pop = Pop(config_path=CONFIG_FILE, fitness_function=eval_line_follower_fitness)
    pop.run(on_generation_end=make_debug_hook(args.debug))

    save_best_indiv(pop, run_name=RUN_NAME)


if __name__ == "__main__":
    main()
