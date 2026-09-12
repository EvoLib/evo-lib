# SPDX-License-Identifier: MIT
"""Run the built-in persistent Foraging simulation."""

import argparse
from pathlib import Path

from evosim.sims.foraging import ForagingConfig, ForagingSession, ForagingSimulation

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
    simulation = ForagingSimulation(config)
    session = ForagingSession(
        simulation,
        render=args.render,
        output_dir=CONFIG_PATH.parent,
    )

    while simulation.running and session.running:
        session.process_events()
        simulation.step()
        session.update()


if __name__ == "__main__":
    main()
