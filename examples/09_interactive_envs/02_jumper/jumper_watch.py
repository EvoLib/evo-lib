# SPDX-License-Identifier: MIT
"""Watch a trained environment checkpoint with Pygame visualization."""

from evolib.evolib_envs.cli import parse_checkpoint_args
from evolib.evolib_envs.core.checkpoint import load_checkpoint
from evolib.evolib_envs.core.task_registry import load_task
from evolib.evolib_envs.envs import register_builtin_tasks

register_builtin_tasks()

args = parse_checkpoint_args()
checkpoint = load_checkpoint(args.checkpoint)

task = load_task(checkpoint)

task.visualize(
    checkpoint.indiv,
    generation=1,
    every=1,
    title=f"Evolved {checkpoint.env.name} ({checkpoint.env.difficulty})",
)
