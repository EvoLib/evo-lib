# SPDX-License-Identifier: MIT
"""Train an EvoLib population on the LineFollower task."""

from evolib import Indiv, Pop, save_best_indiv
from evolib.evolib_envs.cli import parse_args
from evolib.evolib_envs.envs.line_follower_task import LineFollowerTask

RUN_NAME = "line_follower"
CONFIG_FILE = "config.yaml"

args = parse_args()

pop = Pop(config_path=CONFIG_FILE)

line_task = LineFollowerTask(seed=pop.config.random_seed)


def eval_line_follower_fitness(indiv: Indiv) -> None:
    """Evaluate one individual on one LineFollower episode."""

    reward = line_task.evaluate(indiv)
    indiv.fitness = -reward


def on_generation_end(pop: Pop) -> None:
    """Optionally visualize the current best individual."""

    if not args.debug:
        return

    line_task.visualize(
        pop.best(sort=True),
        generation=pop.generation_num,
    )


pop.set_fitness_function(eval_line_follower_fitness)
pop.run(on_generation_end=on_generation_end)
save_best_indiv(pop, run_name=RUN_NAME)
