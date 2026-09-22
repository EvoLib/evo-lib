# SPDX-License-Identifier: MIT
"""Run the built-in persistent Predator-Prey simulation."""

from evosim.sims.predator_prey import PredatorPreySession, PredatorPreySimulation

simulation = PredatorPreySimulation("simulation.yaml")
session = PredatorPreySession(simulation, render=True)

while simulation.running and session.running:
    simulation.step()
    session.update()
