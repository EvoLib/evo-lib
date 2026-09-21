# 02_predator_prey – Competitive Coevolution in a Persistent World

Two independently evolving populations share one persistent 2D world. Predators
hunt Prey, while Prey are selected for avoiding capture. There are no generations
and no explicit fitness values. Selection emerges from survival, reproduction,
mutation, and direct interaction between the two populations.

For the general EvoSim architecture and its distinction from EvoEnv, see
[`evosim/README.md`](../../../evosim/README.md).

---

## Goal

This example demonstrates **continuous competitive coevolution** in a persistent
environment.

Predators depend on finding and capturing Prey. Successful captures restore
energy and therefore directly affect Predator survival and reproduction.

Prey reproduce periodically and die only when captured. Individuals that avoid
Predators remain in the population longer and therefore retain more opportunities
to reproduce.

The two populations consequently modify each other's selection pressure:

```text
better escape
-> fewer successful captures
-> stronger pressure on Predator behavior

better pursuit
-> more captures
-> stronger pressure on Prey behavior
```

This reciprocal dependency is the central purpose of the example.

The simulation can produce changing pursuit and escape strategies, population
fluctuations, bottlenecks, disengagement, or extinction. These changes are
evidence of coevolutionary dynamics, but they are **not by themselves evidence
of cumulative evolutionary progress**.

---

## What This Example Does Not Demonstrate

Competitive coevolution is not equivalent to optimization against a fixed target.

At time `t`, a strategy is evaluated implicitly against the opponents that happen
to exist at that time. A strategy that performs well against the current
population is not necessarily superior to strategies from earlier points in the
run.

The current example therefore does **not** establish a monotonic relation such as:

```text
g100 > g80 > g60 > g40 > g20
```

where later populations are assumed to be generally more capable than earlier
ones.

Several well-known effects can prevent such an interpretation:

- **Cycling** – strategies can alternate between mutually exploitable
  counter-strategies.
- **Forgetting** – capabilities useful against earlier opponents can disappear
  when they are no longer under selection.
- **Disengagement** – one population can become sufficiently dominant that useful
  reciprocal selection pressure collapses.
- **Bootstrap failure** – coevolution may fail to start when useful variation does
  not create a sufficient initial advantage.

Population change therefore means that evolution is occurring, not necessarily
that general capability is increasing.

---

## Capability-Space Limitation

The example intentionally uses a mostly fixed behavioral interface.

The following are fixed by configuration or implementation:

- sensor type and ray layout,
- number of sensor rays,
- sensor field of view and range,
- movement outputs (`turn`, `throttle`),
- movement physics,
- agent body radius,
- controller dimensions,
- controller connectivity structure.

Only the controller parameters are mutated during reproduction.

The evolutionary process therefore explores a **predefined capability space**.
It can discover different strategies within that space, but it cannot invent a
new sensor modality, add an actuator, change the body plan, or structurally expand
the controller.

This matters when interpreting long-term coevolution. The ability of a system to
continue producing qualitatively new counter-adaptations depends on the degrees
of freedom available to evolution. A model with a fixed and relatively small
capability space has an inherent limit on the kinds of novelty it can generate.

This example should therefore be treated as a demonstration of **competitive
coevolution and reciprocal adaptation**, not as a benchmark for open-ended
evolution or sustained innovation.

---

## Measuring Progress

The built-in metrics describe ecological and evolutionary activity, not general
coevolutionary competence.

They include:

- current Prey population,
- current Predator population,
- cumulative Prey births,
- cumulative Predator births,
- cumulative Predator deaths,
- cumulative captures,
- mean Prey energy,
- mean Predator energy.

These metrics are useful for detecting extinction, population stability,
population turnover, and changes in hunting pressure. They cannot determine
whether a later controller is generally superior to an earlier controller.

A stronger test of cumulative coevolutionary progress would require additional
evaluation, for example:

```text
current Predator vs historical Prey
historical Predator vs current Prey
current populations vs an archive of opponents
```

Cross-play against historical opponents can reveal whether later populations
retain previous capabilities and defeat a broader set of opponents rather than
merely adapting to the current population.

Such evaluation is deliberately outside the scope of this example.

---

## Simulation

The world is continuous and toroidal: crossing one boundary enters from the
opposite side.

Each simulation step is:

```text
calculate actions
-> move Prey and Predators
-> remove starved Predators
-> resolve captures
-> reproduce Prey
-> reproduce Predators
```

Prey and Predators evolve independently. Offspring receive a copied EvoLib
`Indiv`, which is mutated before being added to the corresponding population.

The simulation stops when:

- `max_steps` is reached,
- all Prey are extinct, or
- all Predators are extinct.

---

## Prey

Prey energy is a movement budget, not health.

Prey start with a full energy reserve. Movement consumes energy. Remaining almost
stationary restores energy through grazing, up to `energy_capacity`.

```text
stay almost still -> recover movement energy -> easier to approach
move / flee       -> spend movement energy   -> harder to catch
```

A Prey with zero energy cannot translate until enough energy has been recovered,
but it does not die from energy depletion.

Prey die only when captured by a Predator.

### Reproduction

Prey reproduction is timer-based and independent of energy.

Founders start at randomized timer phases. After each successful birth, the next
delay is sampled around `reproduction_interval_steps` using
`reproduction_interval_jitter_steps`.

Reproduction is also constrained by `max_population`.

When a reproduction timer expires, the individual gets one reproduction opportunity.
If population capacity is available, offspring can be created. If the population is
full, that reproduction opportunity is lost. In both cases, the parent's timer is
restarted for the next reproduction interval.

This prevents reproduction attempts from accumulating while the population is full.
Population losses can therefore persist until later reproduction opportunities occur,
allowing Predator success to affect Prey density instead of being immediately masked by
a backlog of reproduction-ready individuals.

Offspring are placed at a randomized displacement from the parent, bounded by
`offspring_dispersion`.

---

## Predators

Predators use energy for both basic survival and movement.

Each step costs:

```text
basal_cost + movement_cost * distance
```

Predators gain energy only by capturing Prey.

A successful capture:

- removes one Prey,
- adds `prey.energy_value` to Predator energy,
- caps energy at `energy_capacity`,
- starts `feeding_cooldown_steps`.

A Predator dies when its energy reaches zero.

### Reproduction

A Predator can reproduce when:

```text
energy >= energy_capacity
and reproduction_cooldown == 0
```

Because feeding is capped at `energy_capacity`, this normally means that the
Predator must reach full energy.

Reproduction splits the current energy equally between parent and offspring.
Both then start a reproduction cooldown.

Predator reproduction is therefore directly coupled to successful hunting.

---

## Vision

Vision is fixed in this baseline.

| Population | Rays | Field of view | Range |
| ---------- | ---: | ------------: | ----: |
| Prey       |   29 |          360° |   100 |
| Predator   |   19 |           90° |    80 |

The asymmetric sensor configuration gives Prey broad awareness while Predators
use a narrower forward-facing field.

Rays are placed at the center of equal angular bins. A 360-degree field therefore
does not duplicate the first and last ray.

For each ray, the sensor returns the nearest circle intersection represented as a
proximity activation. Objects outside the configured range are ignored.

---

## Observation Space

Prey receive one Predator activation per ray plus normalized movement energy:

```text
29 Predator rays + energy = 30 inputs
```

Predators receive one Prey activation per ray plus normalized energy and feeding
cooldown:

```text
19 Prey rays + energy + feeding cooldown = 21 inputs
```

The configuration validates that the EvoNet input dimensions match these
observation dimensions.

---

## Action Space

Both populations use the same two continuous controller outputs:

| Index | Value      | Meaning                           |
| ----: | ---------- | --------------------------------- |
|     0 | `turn`     | Steering in `[-1.0, 1.0]`        |
|     1 | `throttle` | Movement in `[-1.0, 1.0]`        |

Positive throttle moves forward. Negative throttle moves backward using
`reverse_factor`, so reverse movement is slower than forward movement.

---

## Controller

Both populations use EvoNet controllers with one hidden layer:

```text
Prey:      30 inputs -> 6 hidden neurons -> 2 outputs
Predator:  21 inputs -> 6 hidden neurons -> 2 outputs
```

The default configuration uses:

```yaml
activation: [linear, tanh, tanh]

connectivity:
  scope: adjacent
  density: 1.0
  recurrent: [direct]
```

Controller weights and biases mutate during reproduction. Sensor structure,
network dimensions, and the action interface remain fixed.

---

## Default Configuration

The included `simulation.yaml` uses:

| Parameter | Prey | Predator |
| --- | ---: | ---: |
| Initial population | 60 | 20 |
| Maximum population | 100 | 30 |
| Maximum speed | 2.8 | 2.5 |
| Maximum turn rate | 0.22 | 0.11 |
| Offspring dispersion | 200 | 300 |
| Sensor rays | 29 | 19 |
| Sensor FOV | 360° | 90° |
| Sensor range | 100 | 80 |

The world size is `900 x 600`, and the default run length is `50,000` simulation
steps.

These values are example parameters rather than claims about an optimal or
biologically realistic predator-prey system.

---

## Run

```bash
cd examples/10_evosim/02_predator_prey
python run.py
```

The example enables Pygame visualization with `render=True` in `run.py`. Set it
to `False` for headless execution.

During rendering:

| Key | Action |
| --- | --- |
| `S` | Toggle sensor visualization |
| `ESC` | Quit |

The information panel shows population sizes, births, deaths, captures, mean
energy, and the configured sensor geometry.

---

## Metrics

The session prints a status line at the configured `metrics.interval`.

CSV output is optional. Enable it in `simulation.yaml`:

```yaml
metrics:
  interval: 100
  file: metrics.csv
```

Different random seeds can produce substantially different trajectories.
Experiments should therefore be repeated across multiple seeds.

Observed changes in capture rate, population size, survival, or behavior should
be interpreted as changes in the current ecological and coevolutionary state.
They should not be interpreted as cumulative progress without an independent
cross-generation evaluation.

---

## Interpretation

This example is useful for studying:

- reciprocal selection between two evolving populations,
- emergence and loss of pursuit and escape strategies,
- ecological feedback between behavior and population dynamics,
- cycling and transient counter-adaptation,
- disengagement and extinction,
- sensitivity to sensor asymmetry and ecological parameters.

It is not designed to answer whether competitive coevolution can produce
indefinite or open-ended improvement.

The important distinction is:

```text
coevolutionary change != cumulative progress
```

and, more generally:

```text
the potential for continued evolutionary improvement depends on
the degrees of freedom available to evolution
```

In this baseline, those degrees of freedom are deliberately restricted so that
the interaction remains understandable and reproducible.

---

## Background

Predator-prey pursuit and evasion has been used as a competitive coevolution
problem for several decades. A recurring research question is whether reciprocal
selection produces sustained arms-race progress or instead leads to cycling,
forgetting, disengagement, and other forms of relative adaptation.

EvoSim approaches the problem from an artificial-life perspective. Evolution
takes place continuously in a persistent world through local interaction,
energy, reproduction, mutation, and death rather than through explicit
generation boundaries or direct fitness evaluation.

The present example deliberately separates two questions:

1. **Does reciprocal selection produce changing adaptive behavior?**
   This simulation is intended to explore that question.

2. **Does later evolution represent generally superior or open-ended capability?**
   The current simulation and metrics **are not** sufficient to establish that.

---

## References

This example is related to classical work on predator-prey coevolution and
embodied evolution:

- Miller, G. F. & Cliff, D. (1994). *Co-Evolution of Pursuit and Evasion I:
  Biological and Game-Theoretic Foundations.*
- Nolfi, S. & Floreano, D. (1998). *Coevolving predator and prey robots:
  Do "arms races" arise in artificial evolution?* Artificial Life, 4(4),
  311-335.
- Bredeche, N., Haasdijk, E. & Prieto, A. (2018). *Embodied Evolution in
  Collective Robotics: A Review.* Frontiers in Robotics and AI, 5.
