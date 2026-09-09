# 01_foraging – Persistent Resource Competition

A population of evolved Foragers moves through a shared 2D world, searches for
food, spends energy, reproduces, and dies. There are no generations and no
explicit fitness values; selection is driven by survival and reproduction.

For the general EvoSim architecture and its distinction from EvoEnv, see
[`evosim/README.md`](../../../evosim/README.md).

---

<p align="center">
  <img src="https://raw.githubusercontent.com/EvoLib/evo-lib/main/examples/10_evosim/01i_foraging/foraging.png" alt="Foraging sample" width="512"/>
</p>

---

## Goal

Foragers must obtain enough food to survive and reproduce.

Successful behavior depends on:

* detecting and reaching food
* using energy efficiently
* competing with other Foragers for shared resources

Because food is limited and shared, population size and behavior directly affect
selection pressure.

---

## Simulation

The world is continuous and toroidal: crossing one boundary enters from the
opposite side.

Each simulation step:

```text
calculate actions
-> move and spend energy
-> resolve food consumption
-> remove dead Foragers
-> reproduce eligible Foragers
-> spawn food
```

If several Foragers reach the same food item, the closest eligible Forager
consumes it.

---

## Energy and Reproduction

Movement, turning, and basic survival consume energy. Food restores energy.

A Forager dies when its energy is depleted or its maximum lifetime is reached.

Reproduction requires sufficient energy and a minimum age. The parent pays a
reproduction cost, while the offspring starts with its own energy reserve.

The default settings are:

| Setting                      |  Value |
| ---------------------------- | -----: |
| `initial_energy`             | `55.0` |
| `reproduction_threshold`     | `90.0` |
| `reproduction_cost`          | `34.0` |
| `offspring_energy`           | `28.0` |
| `min_reproduction_age_steps` |   `80` |
| `max_lifetime_steps`         | `5000` |

Offspring receive a copied EvoLib `Indiv` which is mutated before being added to
the population.

---

## Observation Space

The controller receives four values:

| Index | Value           | Meaning                   |
| ----: | --------------- | ------------------------- |
|     0 | `food_sensor_0` | Left food sensor          |
|     1 | `food_sensor_1` | Center food sensor        |
|     2 | `food_sensor_2` | Right food sensor         |
|     3 | `energy`        | Normalized current energy |

Each food sensor has an angle, field of view, and range.

For visible food at distance `d`:

```text
activation = 1 - d / sensor_range
```

Food outside the sensor's field of view or range is ignored.

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

This allows sensor geometry to evolve without changing the number of controller
inputs.

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

The default controller is a small feed-forward EvoNet:

```yaml
controller:
  type: evonet
  dim: [4, 6, 2]
  activation: [linear, tanh, tanh]
```

The four inputs correspond to the three food sensors and current energy. The two
outputs control turning and throttle.

Controller parameters mutate together with the enabled sensor parameters during
reproduction.

---

## Run

Headless:

```bash
python examples/10_evosim/01_foraging/run.py
```

With Pygame visualization:

```bash
python examples/10_evosim/01_foraging/run.py --render
```

Both modes use the same simulation logic and `simulation.yaml`.

During rendering:

| Key   | Action                      |
| ----- | --------------------------- |
| `S`   | Toggle sensor visualization |
| `ESC` | Quit                        |

---

## Metrics

The example writes aggregate population, food, birth, death, energy, and
lifetime statistics to `metrics.csv`.

