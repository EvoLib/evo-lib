# SPDX-License-Identifier: MIT
"""Watch a trained Collector checkpoint with Pygame visualization."""

from evoenv.cli import parse_checkpoint_args
from evoenv.core.checkpoint import load_checkpoint
from evoenv.envs.collector_task import CollectorTask


def main() -> None:
    """Load and visualize one trained Collector checkpoint."""
    args = parse_checkpoint_args()
    checkpoint = load_checkpoint(args.checkpoint)
    task = CollectorTask.from_checkpoint(checkpoint)

    task.visualize(
        checkpoint.indiv,
        generation=1,
        every=1,
        title=f"Evolved {checkpoint.env.name}",
    )


if __name__ == "__main__":
    main()
