# SPDX-License-Identifier: MIT
"""Train an EvoLib population on the Jumper task."""

import argparse

from evoenv.core.checkpoint import EnvCheckpoint, EnvSpec, save_checkpoint
from evoenv.envs.jumper_task import JumperTask

from evolib import Indiv, Pop

ENV_NAME = "jumper"
CONFIG_PATH = "config.yaml"
TASK_CONFIG_PATH = "task.yaml"
CHECKPOINT_PATH = "jumper.pkl"
FRAME_FOLDER = "frames"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a Jumper agent.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Render the current best individual during training.",
    )
    return parser.parse_args()


args = parse_args()

pop = Pop(config_path=CONFIG_PATH)
seed = pop.config.random_seed

jumper_task = JumperTask.from_yaml(TASK_CONFIG_PATH, seed=seed)


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
        params={
            "task_config": jumper_task.task_config.to_yaml_dict(),
        },
    ),
    seed=seed,
)

save_checkpoint(CHECKPOINT_PATH, checkpoint)
print(f"Saved checkpoint to: {CHECKPOINT_PATH}")
