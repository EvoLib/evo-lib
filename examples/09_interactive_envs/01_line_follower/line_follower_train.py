# SPDX-License-Identifier: MIT
"""Train an EvoLib population on the LineFollower task."""

from evolib import Indiv, Pop
from evolib.evolib_envs.cli import parse_env_args
from evolib.evolib_envs.core.checkpoint import EnvCheckpoint, EnvSpec, save_checkpoint
from evolib.evolib_envs.core.difficulty import (
    difficulty_checkpoint_path,
    difficulty_config_path,
)
from evolib.evolib_envs.envs.line_follower_task import LineFollowerTask

ENV_NAME = "linefollower"

args = parse_env_args(description="Train a Line Follower agent.")
config_path = difficulty_config_path(args.difficulty)
checkpoint_path = difficulty_checkpoint_path(ENV_NAME, args.difficulty)

pop = Pop(config_path=str(config_path))
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

best_indiv = pop.best(sort=True)
checkpoint = EnvCheckpoint(
    indiv=best_indiv,
    env=EnvSpec(
        name=ENV_NAME,
        difficulty=args.difficulty,
    ),
    seed=seed,
)

save_checkpoint(checkpoint_path, checkpoint)
print(f"Saved checkpoint to: {checkpoint_path}")
