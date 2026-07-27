# SPDX-License-Identifier: MIT
"""Watch a trained Collector checkpoint with Pygame visualization."""

from evoenv.cli import parse_checkpoint_args
from evoenv.core.checkpoint import load_checkpoint
from evoenv.core.task_registry import load_task
from evoenv.envs.collector_task import register_collector_task


def main() -> None:
    """Load and visualize one trained Collector checkpoint."""
    register_collector_task()

    args = parse_checkpoint_args()
    checkpoint = load_checkpoint(args.checkpoint)
    task = load_task(checkpoint)

    task.visualize(
        checkpoint.indiv,
        generation=1,
        every=1,
        title=f"Evolved {checkpoint.env.name}",
    )


if __name__ == "__main__":
    main()
