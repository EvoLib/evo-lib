# SPDX-License-Identifier: MIT
"""Watch a trained Jumper checkpoint with Pygame visualization."""

from evoenv.cli import parse_checkpoint_args
from evoenv.core.checkpoint import load_checkpoint
from evoenv.envs.jumper_task import JumperTask

args = parse_checkpoint_args()
checkpoint = load_checkpoint(args.checkpoint)

task = JumperTask.from_checkpoint(checkpoint)

task.visualize(
    checkpoint.indiv,
    title=f"Evolved {checkpoint.env.name}",
)
