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
