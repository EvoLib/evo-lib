# EvoSim Examples

This directory contains persistent multi-agent evolutionary simulations built with
EvoSim and EvoLib.

Unlike the episodic environments in [`09_evoenv`](../09_evoenv/), EvoSim runs
populations in a persistent world. Individuals coexist, reproduce during the
simulation, and pass mutated EvoLib parameters to their offspring.

Selection emerges from survival, reproduction, competition, and interaction
rather than from an explicit episodic fitness function.

For the package architecture and the distinction between EvoSim and EvoEnv, see
[`evosim/README.md`](../../evosim/README.md).

---

## When to Use EvoSim

EvoSim is useful when population dynamics and persistent interactions are part of
the evolutionary process.

Typical mechanisms include:

* competition for limited resources,
* continuous birth and death,
* population-dependent resource availability,
* predation and escape,
* spatial competition,
* ecological feedback,
* reciprocal selection between evolving populations.

Behavior and selection pressure can form a feedback loop:

```text
behavior
-> survival / reproduction
-> population or environment changes
-> changed selection pressure
-> new behavior
```

For independent evaluation against a fixed episodic task, use EvoEnv.

---

## Examples

| Example                                 | Main focus                                                 |
| --------------------------------------- | ---------------------------------------------------------- |
| [`01_foraging`](01_foraging/)           | Persistent resource competition and energy-based selection |
| [`02_predator_prey`](02_predator_prey/) | Competitive coevolution and reciprocal selection           |

---

## 01 — Foraging

A population of Foragers shares a persistent 2D world with limited food.

Movement and survival consume energy. Food restores energy, and sufficiently
successful individuals reproduce. Offspring inherit mutated controllers and
selected sensor parameters.

Because food is shared, successful individuals affect the resource availability
and reproductive opportunities of the rest of the population.

The optional poison configuration adds a second resource type and requires agents
to distinguish beneficial from harmful objects.

[Open Foraging](01_foraging/)

---

## 02 — Predator-Prey

Two independently evolving populations share one persistent world.

Predators must find and capture Prey to survive and reproduce. Prey gain more
opportunities to reproduce by avoiding capture. Each population therefore changes
the selection pressure experienced by the other.

The example models competitive coevolution between Predator and Prey, including:

* changing pursuit and escape behavior,
* population fluctuations,
* reciprocal adaptation,
* disengagement,
* bottlenecks,
* extinction.

Changing behavior or population dynamics do not by themselves show cumulative
evolutionary progress.

Later populations are adapted to the opponents present at that time. Cycling,
forgetting, and transient counter-adaptation are therefore possible. Testing
whether later populations are generally more capable than earlier ones requires
historical cross-play or another independent evaluation.

The interaction structure is fixed. Sensor modalities and layout, actions,
movement physics, body properties, and controller structure are predefined while
controller parameters evolve.

[Open Predator-Prey](02_predator_prey/)

---

## EvoEnv vs. EvoSim

| EvoEnv                                 | EvoSim                                          |
| -------------------------------------- | ----------------------------------------------- |
| Episodic tasks                         | Persistent worlds                               |
| Explicit reward and fitness            | Selection through survival and reproduction     |
| Agents usually evaluated independently | Individuals coexist and affect each other       |
| Environment resets between evaluations | World state continues over time                 |
| Evolution outside the environment      | Reproduction and mutation inside the simulation |

Use EvoEnv for clearly defined tasks with explicit evaluation.

Use EvoSim when population dynamics, persistent resources, survival,
reproduction, or interactions between individuals are part of the experiment.

---

## Running the Examples

Foraging:

```bash
cd examples/10_evosim/01_foraging
python run.py
```

Predator-Prey:

```bash
cd examples/10_evosim/02_predator_prey
python run.py
```

Both examples use Pygame visualization by default. Rendering can be disabled in
`run.py` for headless experiments.

---

## Interpreting EvoSim Runs

The current EvoSim simulations have no explicit generation boundaries or fitness
values. Their built-in metrics mainly describe ecological and population
dynamics.

Population growth, resource consumption, capture rates, or changing behavior show
that the system is changing. They do not by themselves establish cumulative
evolutionary progress.

In particular:

```text
evolutionary change != cumulative progress
```

and for competitive coevolution:

```text
adaptation to the current opponent
!=
general superiority over earlier populations
```

The range of adaptations is constrained by the predefined environment,
sensorimotor interface, body model, interaction rules, and controller
representation.

EvoSim focuses on small, inspectable simulations of evolutionary and ecological
dynamics. The examples do not demonstrate open-ended evolution or sustained
evolutionary innovation.

