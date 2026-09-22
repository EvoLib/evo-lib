# 02_predator_prey – Competitive Coevolution in a Persistent World

Two independently evolving populations share a persistent 2D world. Predators
hunt Prey, while avoiding capture increases the reproductive opportunities of
Prey.

There are no explicit generation boundaries or fitness values. Selection results
from survival, reproduction, mutation, and interactions between the two
populations.

For the EvoSim architecture and its distinction from EvoEnv, see
[`evosim/README.md`](../../../evosim/README.md).

---

<p align="center">
  <img src="https://raw.githubusercontent.com/EvoLib/evo-lib/main/examples/10_evosim/02_predator_prey/predator_prey.gif" alt="PredatorPrey sample" width="512"/>
</p>

---

## Goal

The model provides a small persistent system for competitive coevolution.

Predators depend on successful captures for energy, survival, and reproduction.
Prey reproduce periodically and die only when captured. Avoiding Predators
therefore increases the time available for reproduction.

Each population affects the selection pressure on the other:

```text
better escape
-> fewer successful captures
-> stronger pressure on Predator behavior

better pursuit
-> more captures
-> stronger pressure on Prey behavior
```

Runs can show changing pursuit and escape behavior, population fluctuations,
bottlenecks, disengagement, or extinction.

---

## Interpreting Coevolution

Competitive coevolution does not optimize against a fixed target.

At time `t`, individuals interact with the opponents present at that time. A
strategy that performs well against the current population is not necessarily
better than strategies from earlier in the run.

The simulation therefore does not establish a monotonic relation such as:

```text
g100 > g80 > g60 > g40 > g20
```

Possible coevolutionary dynamics include:

* **cycling** – different counter-strategies replace one another,
* **forgetting** – behavior useful against earlier opponents disappears,
* **disengagement** – one population becomes dominant enough that reciprocal
  selection pressure weakens,
* **bootstrap failure** – useful reciprocal adaptation fails to develop,
* **extinction** – one population disappears and the interaction ends.

Population change alone is not evidence of cumulative evolutionary progress.

### Fixed Interaction Structure

The following are fixed by configuration or implementation:

* sensor type and ray layout,
* number of sensor rays,
* sensor field of view and range,
* movement outputs (`turn`, `throttle`),
* movement physics,
* agent body radius,
* controller dimensions,
* controller connectivity structure.

Only controller parameters mutate during reproduction.

Evolution can discover different behaviors and strategies within this
interaction structure, but it cannot introduce a new sensor modality, add an
actuator, change the body plan, or structurally expand the controller.

The range of possible adaptations is therefore constrained by the predefined
environment, sensorimotor interface, body model, and controller representation.

The model does not demonstrate open-ended evolution or sustained evolutionary
innovation.

---

## Measuring Progress

The built-in metrics describe population and ecological dynamics:

* current Prey population,
* current Predator population,
* cumulative Prey births,
* cumulative Predator births,
* cumulative Predator deaths,
* cumulative captures,
* mean Prey energy,
* mean Predator energy.

They can show extinction, population turnover, changes in hunting pressure, and
other run dynamics. They do not determine whether later controllers are
generally superior to earlier ones.

Testing cumulative progress requires additional evaluation, for example:

```text
current Predator vs historical Prey
historical Predator vs current Prey
current populations vs an archive of opponents
```

Historical cross-play can test whether later populations retain earlier
capabilities and perform against a broader range of opponents. It is not
implemented by this example.

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

* `max_steps` is reached,
* all Prey are extinct, or
* all Predators are extinct.

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

Reproduction is constrained by `max_population`.

When a reproduction timer expires, the individual gets one reproduction
opportunity. If population capacity is available, offspring can be created. If
the population is full, the opportunity is lost. In both cases, the timer is
restarted.

This prevents reproduction attempts from accumulating while the population is
full. Population losses therefore persist until later reproduction
opportunities.

Offspring are placed at a randomized displacement from the parent, bounded by
`offspring_dispersion`.

---

## Predators

Predators use energy for basic survival and movement.

Each step costs:

```text
basal_cost + movement_cost * distance
```

Predators gain energy only by capturing Prey.

A successful capture:

* removes one Prey,
* adds `prey.energy_value` to Predator energy,
* caps energy at `energy_capacity`,
* starts `feeding_cooldown_steps`.

A Predator dies when its energy reaches zero.

### Reproduction

A Predator can reproduce when:

```text
energy >= energy_capacity
and reproduction_cooldown == 0
```

Because feeding is capped at `energy_capacity`, this normally requires full
energy.

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

Prey have broad awareness, while Predators use a narrower forward-facing field.

Rays are placed at the center of equal angular bins. A 360-degree field therefore
does not duplicate the first and last ray.

For each ray, the sensor returns the nearest circle intersection as a proximity
activation. Objects outside the configured range are ignored.

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

Both populations use two continuous controller outputs:

| Index | Value      | Meaning                   |
| ----: | ---------- | ------------------------- |
|     0 | `turn`     | Steering in `[-1.0, 1.0]` |
|     1 | `throttle` | Movement in `[-1.0, 1.0]` |

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

| Parameter            | Prey | Predator |
| -------------------- | ---: | -------: |
| Initial population   |   60 |       20 |
| Maximum population   |  100 |       30 |
| Maximum speed        |  2.8 |      2.5 |
| Maximum turn rate    | 0.22 |     0.11 |
| Offspring dispersion |  200 |      300 |
| Sensor rays          |   29 |       19 |
| Sensor FOV           | 360° |      90° |
| Sensor range         |  100 |       80 |

The world size is `900 x 600`, and the default run length is `50,000` simulation
steps.

These are example parameters, not a biologically realistic predator-prey model.

---

## Run

```bash
cd examples/10_evosim/02_predator_prey
python run.py
```

Pygame visualization is enabled with `render=True` in `run.py`. Set it to
`False` for headless execution.

During rendering:

| Key   | Action                      |
| ----- | --------------------------- |
| `S`   | Toggle sensor visualization |
| `ESC` | Quit                        |

The information panel shows population sizes, births, deaths, captures, mean
energy, and sensor geometry.

---

## Metrics

The session prints a status line at the configured `metrics.interval`.

CSV output is optional:

```yaml
metrics:
  interval: 100
  file: metrics.csv
```

Different random seeds can produce substantially different trajectories.
Experiments should therefore use multiple seeds.

Capture rate, population size, survival, and behavior describe the current
ecological and coevolutionary state. Cumulative progress requires an independent
cross-generation evaluation.

---

## References

* Miller, G. F. & Cliff, D. (1994). *Co-Evolution of Pursuit and Evasion I:
  Biological and Game-Theoretic Foundations.*
* Nolfi, S. & Floreano, D. (1998). *Coevolving predator and prey robots:
  Do "arms races" arise in artificial evolution?* Artificial Life, 4(4),
  311-335.
* Bredeche, N., Haasdijk, E. & Prieto, A. (2018). *Embodied Evolution in
  Collective Robotics: A Review.* Frontiers in Robotics and AI, 5.

