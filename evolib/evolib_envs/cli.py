# SPDX-License-Identifier: MIT
"""Small shared CLI helpers for EvoLib environment examples."""

import argparse


def add_debug_arg(parser: argparse.ArgumentParser) -> None:
    """Add the common debug flag."""

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show debug visualization during training.",
    )


def parse_args() -> argparse.Namespace:
    """Parse standard arguments for environment examples."""

    parser = argparse.ArgumentParser()
    add_debug_arg(parser)
    return parser.parse_args()


def add_linefollower_difficulty_arg(parser: argparse.ArgumentParser) -> None:
    """Add the common LineFollower difficulty argument."""

    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        default="medium",
        help="LineFollower difficulty preset.",
    )


def add_jumper_difficulty_arg(parser: argparse.ArgumentParser) -> None:
    """Add the common Jumper difficulty argument."""

    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        default="medium",
        help="Jumper difficulty preset.",
    )


def parse_linefollower_args() -> argparse.Namespace:
    """Parse common arguments for Jumper example scripts."""

    parser = argparse.ArgumentParser()
    add_debug_arg(parser)
    add_linefollower_difficulty_arg(parser)
    return parser.parse_args()


def parse_jumper_args() -> argparse.Namespace:
    """Parse common arguments for Jumper example scripts."""

    parser = argparse.ArgumentParser()
    add_debug_arg(parser)
    add_jumper_difficulty_arg(parser)
    return parser.parse_args()
