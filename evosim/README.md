# EvoSim

EvoSim provides a lightweight structure and built-in persistent multi-agent
simulations for evolutionary experiments with EvoLib.

Unlike EvoEnv, EvoSim does not evaluate one controller in a sequence of isolated
episodes. Multiple individuals coexist in the same world while resources,
population size, birth, and death change continuously.

EvoSim supports experiments in which agent-environment interactions affect survival and reproduction.

## Scope

Typical EvoSim experiments may involve:

* competition for limited resources
* continuous birth and death
* energy-based survival and reproduction
* mutation during reproduction
* evolving controllers or sensor parameters
* population-dependent selection pressure
* persistent spatial interaction between individuals

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
| Simulation session        | Metrics output and optional visualization        |
| Renderer                  | Optional visualization                           |


## Built-in Simulations

### Foraging

Foraging is a persistent 2D simulation in which evolved agents compete for a
shared food supply.

Movement consumes energy, food restores it, and sufficiently successful
Foragers reproduce. Offspring inherit mutated EvoLib parameters, including the
controller and selected sensor properties.

There is no explicit fitness function or generation boundary.

[Foraging example](https://github.com/EvoLib/evo-lib/tree/main/examples/10_evosim/01_foraging)

### Predator-Prey

Predator-Prey contains two independently evolving populations. Prey balance
stationary grazing against movement and escape, while Predators must find and
capture Prey to survive and reproduce.

The two populations deliberately use different life-history rules: Prey die only
through predation and reproduce on a time-based schedule, while Predator survival and
reproduction depend on energy gained from successful captures.

[Predator-Prey example](https://github.com/EvoLib/evo-lib/tree/main/examples/10_evosim/02_predator_prey)

## Configuration

Built-in simulations use YAML configuration.

## Headless and Rendered Runs

Simulation logic is independent of visualization. Rendering is controlled when
a simulation session is created:

```python
session = ForagingSession(simulation, render=True)
```

Set `render=False` for headless execution.
