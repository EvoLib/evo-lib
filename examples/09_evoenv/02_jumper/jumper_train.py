# SPDX-License-Identifier: MIT
"""Train an EvoLib population on the Jumper task."""

from evoenv.cli import parse_env_args
from evoenv.core.checkpoint import EnvCheckpoint, EnvSpec, save_checkpoint
from evoenv.core.difficulty import (
    difficulty_checkpoint_path,
    difficulty_config_path,
)
from evoenv.envs.jumper_task import JumperTask

from evolib import Indiv, Pop

ENV_NAME = "jumper"
FRAME_FOLDER = "frames"

args = parse_env_args(description="Train a Jumper agent.")
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
        filename=f"{FRAME_FOLDER}/gen_{pop.generation_num:03d}.gif",
        frame_skip=2,
        gif_fps=30,
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
