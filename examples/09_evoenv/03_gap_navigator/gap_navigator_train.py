# SPDX-License-Identifier: MIT
"""Train an EvoLib population on the GapNavigator task."""

from evoenv.cli import parse_env_args
from evoenv.core.checkpoint import EnvCheckpoint, EnvSpec, save_checkpoint
from evoenv.core.difficulty import (
    difficulty_checkpoint_path,
    difficulty_config_path,
    difficulty_task_path,
)
from evoenv.envs.gap_navigator_task import GapNavigatorTask

from evolib import Indiv, Pop

ENV_NAME = "gap_navigator"
FRAME_FOLDER = "frames"

args = parse_env_args(description="Train an GapNavigator agent.")


config_path = difficulty_config_path(args.difficulty)
task_config_path = difficulty_task_path(args.difficulty)
checkpoint_path = difficulty_checkpoint_path(ENV_NAME, args.difficulty)

pop = Pop(config_path=str(config_path))
seed = pop.config.random_seed

obstacle_task = GapNavigatorTask.from_yaml(
    task_config_path,
    seed=seed,
)


def eval_gap_navigator_fitness(indiv: Indiv) -> None:
    """Evaluate one individual on one GapNavigator episode."""
    sensor_length_scale = obstacle_task.fitness_config.sensor_length_scale
    sensor_length_penalty = obstacle_task.fitness_config.sensor_length_penalty
    sensor_count_penalty = obstacle_task.fitness_config.sensor_count_penalty
    min_active_length = obstacle_task.sensor_config.min_active_length

    reward = obstacle_task.evaluate(indiv)

    sensor_layout = obstacle_task.make_sensor_layout(indiv)

    active_sensor_count = sum(
        sensor.length > 0.0 and sensor.length >= min_active_length
        for sensor in sensor_layout
    )

    total_sensor_length = sum(sensor.length for sensor in sensor_layout)

    sensor_penalty = (
        active_sensor_count * sensor_count_penalty
        + (total_sensor_length / sensor_length_scale) * sensor_length_penalty
    )

    indiv.fitness = -(reward - sensor_penalty)


def on_generation_end(pop: Pop) -> None:
    """Optionally visualize the current best individual."""
    if not args.debug:
        return

    obstacle_task.visualize(
        pop.best(sort=True),
        generation=pop.generation_num,
        filename=f"{FRAME_FOLDER}/gen_{pop.generation_num:03d}.gif",
        frame_skip=2,
        gif_fps=30,
    )


pop.set_fitness_function(eval_gap_navigator_fitness)
pop.run(on_generation_end=on_generation_end)

best_indiv = pop.best(sort=True)
checkpoint = EnvCheckpoint(
    indiv=best_indiv,
    env=EnvSpec(
        name=ENV_NAME,
        difficulty=args.difficulty,
        params={
            "task_config": obstacle_task.task_config.to_yaml_dict(),
        },
    ),
    seed=seed,
)

save_checkpoint(checkpoint_path, checkpoint)
print(f"Saved checkpoint to: {checkpoint_path}")
