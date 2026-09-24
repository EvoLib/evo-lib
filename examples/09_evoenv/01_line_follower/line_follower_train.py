# SPDX-License-Identifier: MIT
"""Train an EvoLib population on the LineFollower task."""

from evoenv.cli import parse_difficulty_args
from evoenv.core.checkpoint import EnvCheckpoint, EnvSpec, save_checkpoint
from evoenv.envs.line_follower_task import LineFollowerTask

from evolib import Indiv, Pop

ENV_NAME = "line_follower"
FRAME_FOLDER = "frames"
DEBUG = True
DEBUG_EVERY = 5

args = parse_difficulty_args(description="Train a Line Follower agent.")
difficulty = args.difficulty
config_path = f"config_{difficulty}.yaml"
task_config_path = f"task_{difficulty}.yaml"
checkpoint_path = f"{ENV_NAME}_{difficulty}.pkl"

pop = Pop(config_path=str(config_path))
seed = pop.config.random_seed

line_task = LineFollowerTask.from_yaml(
    task_config_path,
    seed=seed,
)


def eval_line_follower_fitness(indiv: Indiv) -> None:
    """Evaluate one individual on one LineFollower episode."""
    reward = line_task.evaluate(indiv)
    indiv.fitness = -reward


def on_generation_end(pop: Pop) -> None:
    """Optionally visualize the current best individual."""
    if DEBUG and pop.generation_num % DEBUG_EVERY == 0:
        line_task.visualize(
            pop.best(sort=True),
            title=f"Training Debug - Gen {pop.generation_num}",
            filename=f"{FRAME_FOLDER}/gen_{pop.generation_num:03d}.gif",
            frame_skip=2,
            gif_fps=30,
        )


pop.set_fitness_function(eval_line_follower_fitness)
pop.run(on_generation_end=on_generation_end)

best_indiv = pop.best(sort=True)
checkpoint = EnvCheckpoint(
    indiv=best_indiv,
    env=EnvSpec(
        name=ENV_NAME,
        difficulty=difficulty,
        params={
            "task_config": line_task.task_config.to_yaml_dict(),
        },
    ),
    seed=seed,
)

save_checkpoint(checkpoint_path, checkpoint)
print(f"Saved checkpoint to: {checkpoint_path}")
