# SPDX-License-Identifier: MIT
"""Run the built-in persistent Foraging simulation."""

from evosim.sims.foraging import ForagingSession, ForagingSimulation

simulation = ForagingSimulation("simulation.yaml")
session = ForagingSession(simulation, render=True)

while simulation.running and session.running:
    simulation.step()
    session.update()
