# SPDX-License-Identifier: MIT
"""Train an EvoLib population on the LineFollower task."""

from evolib import Indiv, Pop, save_best_indiv
from evolib.evolib_envs.cli import parse_linefollower_args
from evolib.evolib_envs.envs.line_follower_task import LineFollowerTask

CONFIG_BY_DIFFICULTY = {
    "easy": "config_easy.yaml",
    "medium": "config_medium.yaml",
    "hard": "config_hard.yaml",
}

BEST_BY_DIFFICULTY = {
    "easy": "best_linefollower_easy.pkl",
    "medium": "best_linefollower_medium.pkl",
    "hard": "best_linefollower_hard.pkl",
}

args = parse_linefollower_args()

config_path = CONFIG_BY_DIFFICULTY[args.difficulty]
best_path = BEST_BY_DIFFICULTY[args.difficulty]

pop = Pop(config_path=config_path)
seed = pop.config.random_seed

line_task = LineFollowerTask(seed=seed, difficulty=args.difficulty)


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

save_best_indiv(pop, run_name=best_path)
print(f"Saved best individual to: {best_path}")
