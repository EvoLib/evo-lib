# SPDX-License-Identifier: MIT
"""Small shared CLI helpers for EvoLib environment examples."""

import argparse
from pathlib import Path
from typing import Sequence

DEFAULT_DIFFICULTIES: tuple[str, ...] = ("easy", "medium", "hard")


def add_debug_arg(parser: argparse.ArgumentParser) -> None:
    """Add the common debug flag."""

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show debug visualization during training.",
    )


def add_difficulty_arg(
    parser: argparse.ArgumentParser,
    *,
    choices: Sequence[str] = DEFAULT_DIFFICULTIES,
    default: str = "medium",
    help_text: str = "Environment difficulty preset.",
) -> None:
    """Add a generic difficulty argument."""

    parser.add_argument(
        "--difficulty",
        choices=choices,
        default=default,
        help=help_text,
    )


def parse_env_args(
    *,
    description: str | None = None,
    with_debug: bool = True,
    with_difficulty: bool = True,
    difficulty_choices: Sequence[str] = DEFAULT_DIFFICULTIES,
    default_difficulty: str = "medium",
) -> argparse.Namespace:
    """Parse common arguments for environment example scripts."""

    parser = argparse.ArgumentParser(description=description)

    if with_debug:
        add_debug_arg(parser)

    if with_difficulty:
        add_difficulty_arg(
            parser,
            choices=difficulty_choices,
            default=default_difficulty,
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
