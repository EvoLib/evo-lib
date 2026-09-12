# SPDX-License-Identifier: MIT
"""Run the built-in persistent Foraging simulation."""

from pathlib import Path

from evosim.cli import parse_sim_args
from evosim.sims.foraging import ForagingConfig, ForagingSession, ForagingSimulation

CONFIG_PATH = Path(__file__).with_name("simulation.yaml")


def main() -> None:
    args = parse_sim_args(description=__doc__)

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
