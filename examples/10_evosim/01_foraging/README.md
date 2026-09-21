# 01_foraging – Persistent Resource Competition

A population of evolved Foragers moves through a shared 2D world, searches for
food, spends energy, reproduces, and dies. There are no generations and no
explicit fitness values; selection is driven by survival and reproduction.

`simulation.yaml` is food-only. `simulation_poison.yaml` enables 
poison variant.

For the general EvoSim architecture and its distinction from EvoEnv, see
[`evosim/README.md`](../../../evosim/README.md).

---

<p align="center">
  <img src="https://raw.githubusercontent.com/EvoLib/evo-lib/main/examples/10_evosim/01_foraging/foraging.gif" alt="Foraging sample" width="512"/>
</p>

---

## Goal

In the baseline, Foragers must obtain enough food to survive and reproduce.

Successful behavior depends on:

* detecting and reaching food
* using energy efficiently
* competing with other Foragers for shared resources

Because food is limited and shared, population size and behavior directly affect
selection pressure.

The poison variant adds a second consumable object type. Poison reduces energy,
so successful Foragers must distinguish positive from negative resources.

---

## Simulation

The world is continuous and toroidal: crossing one boundary enters from the
opposite side.

The baseline simulation step is:

```text
calculate actions
-> move and spend energy
-> remove dead Foragers
-> resolve food consumption
-> reproduce eligible Foragers
-> spawn food
```

With poison enabled, poison consumption is resolved after food consumption. A
Forager killed by poison is removed before reproduction, and new poison is
spawned after food.

If several Foragers reach the same food or poison item, the closest eligible
Forager gets it. A Forager already at its energy capacity or still in its
feeding cooldown does not consume food.

---

## Energy and Reproduction

Movement, turning, and basic survival consume energy. Food restores energy and
starts a short feeding cooldown during which no additional food can be consumed.
The remaining cooldown is available to the controller as a normalized internal
state. In the poison variant, poison removes a fixed amount of energy.

A Forager dies when its energy is depleted or its maximum age is reached.

Reproduction requires sufficient energy and a minimum age. The parent pays a
reproduction cost, while the offspring starts with its own energy reserve.

The default settings are:

| Setting                      |  Value |
| ---------------------------- | -----: |
| `initial_energy`             | `55.0` |
| `feeding_cooldown_steps`     |   `20` |
| `reproduction_threshold`     | `90.0` |
| `reproduction_cost`          | `34.0` |
| `offspring_energy`           | `28.0` |
| `min_reproduction_age_steps` |   `80` |
| `max_age_steps`              | `5000` |

Offspring receive a copied EvoLib `Indiv` which is mutated before being added to
the population.

---

## Observation Space

With poison disabled, the controller receives five values:

| Index | Value              | Meaning                              |
| ----: | ------------------ | ------------------------------------ |
|     0 | `food_sensor_0`    | Left food sensor                     |
|     1 | `food_sensor_1`    | Center food sensor                   |
|     2 | `food_sensor_2`    | Right food sensor                    |
|     3 | `energy`           | Normalized current energy            |
|     4 | `feeding_cooldown` | Normalized remaining feeding cooldown |

With poison enabled, the same three spatial sensors each produce separate food
and poison channels:

| Index | Value             | Meaning                         |
| ----: | ----------------- | ------------------------------- |
|     0 | `sensor_0_food`   | Food signal in the left sensor  |
|     1 | `sensor_0_poison` | Poison signal in the left sensor |
|     2 | `sensor_1_food`   | Food signal in the center sensor |
|     3 | `sensor_1_poison` | Poison signal in the center sensor |
|     4 | `sensor_2_food`   | Food signal in the right sensor |
|     5 | `sensor_2_poison` | Poison signal in the right sensor |
|     6 | `energy`          | Normalized current energy       |
|     7 | `feeding_cooldown` | Normalized remaining feeding cooldown |

Each spatial sensor has one angle, field of view, and range. For a visible
resource at distance `d`:

```text
activation = 1 - d / sensor_range
```

Objects outside the sensor's field of view or range are ignored.

---

## Sensor Evolution

All founders start with the same three-sensor geometry:

```yaml
sensor_angles:
  values: [-0.6, 0.0, 0.6]

sensor_fovs:
  values: [0.7, 0.7, 0.7]

sensor_ranges:
  values: [120.0, 120.0, 120.0]
```

Sensor angles remain fixed, while field of view and range mutate in offspring.
The poison variant reuses exactly the same geometry for both sensory channels.

---

## Action Space

The controller returns two continuous outputs:

| Index | Value      | Meaning                                      |
| ----: | ---------- | -------------------------------------------- |
|     0 | `turn`     | Steering in `[-1.0, 1.0]`                    |
|     1 | `throttle` | Forward or reverse movement in `[-1.0, 1.0]` |

Reverse movement is possible but slower than forward movement.

---

## Controller

The baseline controller uses five inputs:

```yaml
controller:
  type: evonet
  dim: [5, 6, 2]
```

The poison variant uses eight inputs:

```yaml
controller:
  type: evonet
  dim: [8, 6, 2]
```

The two outputs control turning and throttle. Controller parameters mutate
together with the enabled sensor parameters during reproduction.

---

## Run

Run the food-only baseline:

```bash
cd examples/10_evosim/01_foraging
python run.py
```

Run the simulation:

```bash
python run.py
```

The example enables Pygame visualization with `render=True` in `run.py`. Set it
to `False` for headless execution. Both configurations use the same simulation
logic.

During rendering:

| Key   | Action                      |
| ----- | --------------------------- |
| `S`   | Toggle sensor visualization |
| `ESC` | Quit                        |

---

## Metrics

The session prints a compact status line at the configured `metrics.interval`.
The poison metrics remain zero when poison is disabled.

CSV output is optional. Set `metrics.file` to a path such as `metrics.csv` to
write the complete metric set. If `file` is omitted or `null`, no CSV file is
created.

