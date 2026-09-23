# EvoSim Examples

This directory contains persistent multi-agent evolutionary simulations built with
EvoSim and EvoLib.

Unlike the episodic environments in [`09_evoenv`](../09_evoenv/), EvoSim runs
populations in a persistent world. Individuals coexist, reproduce during the
simulation, and pass mutated EvoLib parameters to their offspring.

Selection emerges from survival, reproduction, competition, and interaction
rather than from an explicit episodic fitness function.

The package architecture and the distinction between EvoSim and EvoEnv are
documented in [`evosim/README.md`](../../evosim/README.md).

---

## 01 — Foraging

<p align="center">
  <img src="https://raw.githubusercontent.com/EvoLib/evo-lib/main/examples/10_evosim/01_foraging/foraging.gif" alt="Foraging sample" width="512"/>
</p>

A population of Foragers shares a persistent 2D world with limited resources.
Movement and survival consume energy, food restores energy, and successful
individuals reproduce.

Because resources are shared, individual behavior affects resource availability
and therefore the selection pressure experienced by the population. An optional
poison configuration adds a second resource type that agents must distinguish
from food.

[Open Foraging](01_foraging/)

---

## 02 — Predator-Prey

<p align="center">
  <img src="https://raw.githubusercontent.com/EvoLib/evo-lib/main/examples/10_evosim/02_predator_prey/predator_prey.gif" alt="Predator-Prey sample" width="512"/>
</p>

Two independently evolving populations share one persistent world. Predators
depend on successful captures for survival and reproduction, while avoiding
capture increases the reproductive opportunities of Prey.

Each population therefore changes the selection pressure experienced by the
other. Runs can show pursuit and escape behavior, population fluctuations,
reciprocal adaptation, bottlenecks, disengagement, or extinction.

Changes in population dynamics or behavior do not by themselves demonstrate
cumulative evolutionary progress. The example README discusses these limitations
and possible evaluation methods in more detail.

[Open Predator-Prey](02_predator_prey/)

