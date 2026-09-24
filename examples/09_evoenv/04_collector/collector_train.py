# SPDX-License-Identifier: MIT
"""Train an EvoLib population on the Collector task."""

from evoenv.core.checkpoint import EnvCheckpoint, EnvSpec, save_checkpoint
from evoenv.envs.collector_task import CollectorTask

from evolib import Indiv, Pop

ENV_NAME = "collector"
CONFIG_PATH = "config.yaml"
TASK_CONFIG_PATH = "task.yaml"
CHECKPOINT_PATH = "collector.pkl"
FRAME_FOLDER = "frames"
DEBUG = True
DEBUG_EVERY = 5


def main() -> None:
    """Train a Collector controller and save the best individual."""
    pop = Pop(config_path=CONFIG_PATH)
    seed = pop.config.random_seed
    collector_task = CollectorTask.from_yaml(TASK_CONFIG_PATH, seed=seed)

    def eval_collector_fitness(indiv: Indiv) -> None:
        """Evaluate one individual on one Collector episode."""
        reward = collector_task.evaluate(indiv)
        indiv.fitness = -reward

    def on_generation_end(current_pop: Pop) -> None:
        """Optionally visualize the current best individual."""
        if DEBUG and current_pop.generation_num % DEBUG_EVERY == 0:
            collector_task.visualize(
                current_pop.best(sort=True),
                title=f"Collector Training Debug - Gen {current_pop.generation_num}",
                filename=f"{FRAME_FOLDER}/gen_{current_pop.generation_num:03d}.gif",
                frame_skip=2,
                gif_fps=30,
            )

    pop.set_fitness_function(eval_collector_fitness)
    pop.run(on_generation_end=on_generation_end)

    best_indiv = pop.best(sort=True)
    checkpoint = EnvCheckpoint(
        indiv=best_indiv,
        env=EnvSpec(
            name=ENV_NAME,
            params={
                "task_config": collector_task.task_config.to_yaml_dict(),
                "module": collector_task.module,
            },
        ),
        seed=seed,
    )

    save_checkpoint(CHECKPOINT_PATH, checkpoint)
    print(f"Saved checkpoint to: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
