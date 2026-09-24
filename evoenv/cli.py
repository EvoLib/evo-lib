# SPDX-License-Identifier: MIT
"""Small shared CLI helpers for EvoLib environment examples."""

import argparse
from pathlib import Path

DEFAULT_DIFFICULTIES: tuple[str, ...] = ("easy", "medium", "hard")


def parse_difficulty_args(
    *,
    description: str | None = None,
) -> argparse.Namespace:
    """Parse the shared difficulty argument for example scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--difficulty",
        choices=DEFAULT_DIFFICULTIES,
        default="medium",
        help="Environment difficulty preset.",
    )
    return parser.parse_args()


def parse_checkpoint_args(
    *,
    description: str = "Watch a trained environment checkpoint.",
) -> argparse.Namespace:
    """Parse CLI arguments for checkpoint-based scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to a trained checkpoint file.",
    )
    return parser.parse_args()
