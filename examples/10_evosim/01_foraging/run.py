# SPDX-License-Identifier: MIT
"""Run the built-in persistent Foraging simulation."""

import argparse
from pathlib import Path

from evosim.sims.foraging import ForagingConfig, run_foraging

CONFIG_PATH = Path(__file__).with_name("simulation.yaml")


def main() -> None:
    """Run Foraging from its YAML configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--render",
        action="store_true",
        help="show the Pygame visualization; headless execution is the default",
    )
    args = parser.parse_args()

    config = ForagingConfig.from_yaml(CONFIG_PATH)
    run_foraging(config, render=args.render, output_dir=CONFIG_PATH.parent)


if __name__ == "__main__":
    main()
