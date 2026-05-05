# SPDX-License-Identifier: MIT
"""Train an EvoLib population on the Jumper task."""

from evolib import Indiv, Pop
from evolib.evolib_envs.cli import parse_jumper_args
from evolib.evolib_envs.core.checkpoint import EnvCheckpoint, EnvSpec, save_checkpoint
from evolib.evolib_envs.core.difficulty import (
    difficulty_checkpoint_path,
    difficulty_config_path,
)
from evolib.evolib_envs.envs.jumper_task import JumperTask

ENV_NAME = "jumper"

args = parse_jumper_args()
config_path = difficulty_config_path(args.difficulty)
checkpoint_path = difficulty_checkpoint_path(ENV_NAME, args.difficulty)

pop = Pop(config_path=str(config_path))
seed = pop.config.random_seed

jumper_task = JumperTask(seed=seed, difficulty=args.difficulty)


def eval_jumper_fitness(indiv: Indiv) -> None:
    """Evaluate one individual on one Jumper episode."""

    reward = jumper_task.evaluate(indiv)
    indiv.fitness = -reward


def on_generation_end(pop: Pop) -> None:
    """Optionally visualize the current best individual."""

    if not args.debug:
        return

    jumper_task.visualize(
        pop.best(sort=True),
        generation=pop.generation_num,
    )


pop.set_fitness_function(eval_jumper_fitness)
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
