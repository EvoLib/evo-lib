# SPDX-License-Identifier: MIT
import argparse


def parse_sim_args(*, description: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--render",
        action="store_true",
        help="Show visualization.",
    )
    return parser.parse_args()
