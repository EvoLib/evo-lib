# SPDX-License-Identifier: MIT
"""Train an EvoLib population on the pixel-based LineFollowerEnv."""

import argparse
from typing import Callable

from evolib import Indiv, Pop, save_best_indiv
from evolib.evolib_envs.cli import add_debug_arg
from evolib.evolib_envs.core.env import Action, Observation
from evolib.evolib_envs.core.evaluator import evaluate_episode
from evolib.evolib_envs.envs.line_follower import LineFollowerEnv
from evolib.evolib_envs.renderers.pygame_line_follower import (
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    run_debug_episode,
)

RUN_NAME = "line_follower"
CONFIG_FILE = "config.yaml"

ENV_WIDTH = SCREEN_WIDTH
ENV_HEIGHT = SCREEN_HEIGHT
MAX_STEPS = 1400
SEED = 42

DEBUG_EVERY_N_GENERATIONS = 5
DEBUG_MAX_STEPS = 400


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()
    add_debug_arg(parser)
    return parser.parse_args()


class LineFollowerController:
    """Interpret EvoNet output as a steering action."""

    def __init__(self, indiv: Indiv) -> None:
        self.net = indiv.para["brain"]

    def act(self, observation: Observation) -> Action:
        """Return a clipped steering action in [-1, 1]."""

        output = self.net.calc(observation)
        turn = float(output[0])
        turn = max(-1.0, min(1.0, turn))
        return [turn]


def make_env() -> LineFollowerEnv:
    """Create the training environment with the same size as the renderer."""

    return LineFollowerEnv(
        width=ENV_WIDTH,
        height=ENV_HEIGHT,
        max_steps=MAX_STEPS,
    )


def eval_line_follower_fitness(indiv: Indiv) -> None:
    """Evaluate one individual on one LineFollower episode."""

    env = make_env()
    controller = LineFollowerController(indiv)

    total_reward = evaluate_episode(
        env,
        controller,
        seed=SEED,
        max_steps=MAX_STEPS,
    )

    # EvoLib currently minimizes fitness values.
    indiv.fitness = -total_reward


def make_debug_hook(debug_enabled: bool) -> Callable[[Pop], None]:
    """Create a generation-end hook for optional debug rendering."""

    def on_generation_end(pop: Pop) -> None:
        best = pop.best(sort=True)

        run_debug_episode(
            make_env(),
            LineFollowerController(best),
            enabled=debug_enabled,
            generation=pop.generation_num,
            every=DEBUG_EVERY_N_GENERATIONS,
            steps=DEBUG_MAX_STEPS,
            seed=SEED,
            title=f"Training Debug - Gen {pop.generation_num}",
        )

    return on_generation_end


def main() -> None:
    """Train the population and save the best individual."""

    args = parse_args()

    pop = Pop(config_path=CONFIG_FILE, fitness_function=eval_line_follower_fitness)
    pop.run(on_generation_end=make_debug_hook(args.debug))

    save_best_indiv(pop, run_name=RUN_NAME)


if __name__ == "__main__":
    main()
