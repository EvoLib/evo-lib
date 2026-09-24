# SPDX-License-Identifier: MIT
"""Train an EvoLib population on the Jumper task."""

from evoenv.core.checkpoint import EnvCheckpoint, EnvSpec, save_checkpoint
from evoenv.envs.jumper_task import JumperTask

from evolib import Indiv, Pop

ENV_NAME = "jumper"
CONFIG_PATH = "config.yaml"
TASK_CONFIG_PATH = "task.yaml"
CHECKPOINT_PATH = "jumper.pkl"
FRAME_FOLDER = "frames"
DEBUG = True
DEBUG_EVERY = 5

pop = Pop(config_path=CONFIG_PATH)
seed = pop.config.random_seed

jumper_task = JumperTask.from_yaml(TASK_CONFIG_PATH, seed=seed)


def eval_jumper_fitness(indiv: Indiv) -> None:
    """Evaluate one individual on one Jumper episode."""
    reward = jumper_task.evaluate(indiv)
    indiv.fitness = -reward


def on_generation_end(pop: Pop) -> None:
    """Optionally visualize the current best individual."""
    if DEBUG and pop.generation_num % DEBUG_EVERY == 0:
        jumper_task.visualize(
            pop.best(sort=True),
            title=f"Jumper Training Debug - Gen {pop.generation_num}",
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
        params={
            "task_config": jumper_task.task_config.to_yaml_dict(),
        },
    ),
    seed=seed,
)

save_checkpoint(CHECKPOINT_PATH, checkpoint)
print(f"Saved checkpoint to: {CHECKPOINT_PATH}")
