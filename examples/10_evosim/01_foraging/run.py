# SPDX-License-Identifier: MIT
"""Run the built-in persistent Foraging simulation."""

from evosim.cli import parse_sim_args
from evosim.sims.foraging import ForagingSession, ForagingSimulation


def main() -> None:
    args = parse_sim_args(description=__doc__)

    simulation = ForagingSimulation("simulation.yaml")
    session = ForagingSession(simulation, render=args.render)

    while simulation.running and session.running:
        session.process_events()
        simulation.step()
        session.update()


if __name__ == "__main__":
    main()
