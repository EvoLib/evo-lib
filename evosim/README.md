# EvoSim

EvoSim provides small, persistent multi-agent simulations for evolutionary
experiments with EvoLib.

Unlike EvoEnv, EvoSim does not evaluate one controller in a sequence of isolated
episodes. Multiple individuals coexist in the same world while resources,
population size, birth, and death change continuously.

This makes EvoSim suitable for experiments where selection pressure emerges from
the interaction between agents and their environment.

## Scope

Typical EvoSim experiments may involve:

* competition for limited resources
* continuous birth and death
* energy-based survival and reproduction
* mutation during reproduction
* evolving controllers or sensor parameters
* population-dependent selection pressure
* persistent spatial interaction between individuals

EvoSim is intended for small, inspectable research and demonstration
simulations rather than large-scale agent-based simulation.

## EvoSim and EvoEnv

EvoEnv and EvoSim cover different experimental setups.

| EvoEnv                                            | EvoSim                                                     |
| ------------------------------------------------- | ---------------------------------------------------------- |
| Episodic                                          | Persistent                                                 |
| One controller evaluated per environment instance | Many individuals share one world                           |
| Explicit reward and fitness                       | Selection is driven by survival and reproduction           |
| Environment resets between evaluations            | World state continues over time                            |
| Evolution happens outside the environment         | Reproduction and mutation happen inside the simulation     |

Use EvoEnv when a controller should solve a clearly defined episodic task.
Use EvoSim when the population and the world should be part of the evolutionary
process.

## Structure

| Component                 | Responsibility                                   |
| ------------------------- | ------------------------------------------------ |
| `Simulation`              | Minimal `reset()` / `step()` interface           |
| Simulation implementation | World state and interaction rules                |
| Objects                   | Simulation-specific agents, resources, and state |
| Configuration             | Validated simulation parameters                  |
| Renderer                  | Optional visualization                           |


## Built-in Simulations

### Foraging

Foraging is a persistent 2D simulation in which evolved agents compete for a
shared food supply.

Movement consumes energy, food restores it, and sufficiently successful
Foragers reproduce. Offspring inherit mutated EvoLib parameters, including the
controller and selected sensor properties.

There is no explicit fitness function or generation boundary.

See
[`examples/10_evosim/01_foraging/`](../examples/10_evosim/01_foraging/).

## Configuration

Built-in simulations use YAML configuration.

## Headless and Rendered Runs

Simulation logic is independent of visualization. The same simulation can run
headless or with a renderer for inspection.

For example:

```bash
python examples/10_evosim/01_foraging/run.py
python examples/10_evosim/01_foraging/run.py --render
```
