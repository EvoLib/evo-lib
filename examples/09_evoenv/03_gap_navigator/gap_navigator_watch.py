# SPDX-License-Identifier: MIT
"""Watch a trained GapNavigator checkpoint with Pygame visualization."""

from evoenv.cli import parse_checkpoint_args
from evoenv.core.checkpoint import load_checkpoint
from evoenv.envs.gap_navigator_task import GapNavigatorTask

args = parse_checkpoint_args()
checkpoint = load_checkpoint(args.checkpoint)

task = GapNavigatorTask.from_checkpoint(checkpoint)

task.visualize(
    checkpoint.indiv,
    title=f"Evolved {checkpoint.env.name} ({checkpoint.env.difficulty})",
)
