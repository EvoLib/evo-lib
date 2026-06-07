# 03_gap_navigator – Gap Steering with Evolvable Sensors

This example demonstrates how an evolved controller can steer through falling
obstacle rows with free gaps.

Compared with Jumper, GapNavigator uses continuous horizontal steering instead
of a discrete jump action. The controller does not receive the gap position
directly. It must steer based on ray sensor values and its own horizontal state.

This example focuses on:

- continuous horizontal steering
- evolved ray sensor layouts
- obstacle avoidance from proximity signals
- comparing rule-based and evolved controllers

For a general overview of the Pygame-based EvoEnv examples, see the main
README in `examples/09_evoenv/`.

---

<p align="center">
  <img src="https://raw.githubusercontent.com/EvoLib/evo-lib/main/examples/09_evoenv/03_gap_navigator/frames/gap_navigator.gif" alt="GapNavigator sample" width="512"/>
</p>

---

## Goal

The player should move horizontally and pass through the gaps in falling
obstacle rows.

The task is solved well when the controller:

- reacts to approaching obstacle blocks
- moves toward open gaps early enough
- avoids oversteering
- avoids staying near the screen edges
- passes many obstacle rows without collisions
- uses useful sensor geometry when sensor penalties are enabled

---

## Task Structure

GapNavigator uses two evolved parameter modules:

| Module | Purpose |
|---|---|
| `sensors` | Encodes ray sensor lengths and angles |
| `brain` | EvoNet controller that maps observations to steering actions |

The sensor module is a flat vector. Each sensor uses two values:

| Vector Position | Meaning |
|---:|---|
| even index | Sensor length, normalized to `[0.0, 1.0]` |
| odd index | Sensor angle, normalized to `[0.0, 1.0]` |

For example, with `max_sensors: 6`, the sensor vector has dimension `12`:

```yaml
modules:
  sensors:
    type: vector
    dim: 12  # 6 lengths + 6 angles
```

The task converts this vector into `RaySensor` objects. Very short sensors can
be represented as zero-length inactive sensors, depending on `min_active_length`.
This keeps the observation size stable even when some sensor slots are inactive.

---

## Observation Space

The environment returns `max_sensors + 2` values.

With the default `max_sensors: 6`, the observation size is `8`.

| Index | Value | Meaning |
|---:|---|---|
| `0..max_sensors-1` | `sensor_value` | Proximity activation for each ray sensor |
| `max_sensors` | `normalized_x` | Horizontal player position normalized to `[0.0, 1.0]` |
| `max_sensors + 1` | `normalized_velocity_x` | Horizontal velocity normalized by player speed |

Typical interpretation:

| Observation Pattern | Meaning |
|---|---|
| sensor value close to `0.0` | no nearby obstacle hit on that ray |
| sensor value close to `1.0` | obstacle hit close to the player |
| `normalized_x` close to `0.0` or `1.0` | player is near a screen edge |
| `normalized_velocity_x < 0.0` / `> 0.0` | player moves left / right |

The controller does not receive the gap center directly. This is intentional:
the gap center is useful for reward calculation and debugging, but not part of
the controller observation.

---

## Action Space

The controller returns one value.

| Index | Value | Meaning |
|---:|---|---|
| 0 | `steering` | Horizontal steering clipped to `[-1.0, 1.0]` |

Typical actions:

```python
[-1.0]  # full left
[0.0]   # no horizontal movement
[1.0]   # full right
```

---

## Controller and Sensor Config

The medium training config uses one vector module for sensors and one EvoNet
module for the controller:

```yaml
modules:
  sensors:
    type: vector
    dim: 12

  brain:
    type: evonet
    dim: [8, 12, 1]
    activation: [linear, tanh, tanh]
```

This matches the default medium task setup:

| Component | Size | Meaning |
|---|---:|---|
| sensor vector | 12 | 6 sensor lengths + 6 sensor angles |
| EvoNet input | 8 | 6 sensor values + `normalized_x` + `normalized_velocity_x` |
| EvoNet hidden | 12 | small nonlinear controller layer |
| EvoNet output | 1 | steering action |

---

## Reward

The environment itself returns `0.0` as step reward. The task computes the
training reward from the environment `info` dictionary.

The task-level reward encourages:

- horizontal alignment with the next relevant gap
- avoiding obstacle collisions
- avoiding unnecessary movement
- avoiding positions near screen edges

The default medium task configuration uses shaped feedback:

```yaml
reward:
  pass_reward: 0.0
  gap_alignment_reward: 0.040
  movement_penalty: 0.014
  collision_penalty: 6.5
  near_wall_penalty: 0.040
```

`pass_reward` is only used when `terminate_on_collision` is enabled.

The fitness function can additionally penalize sensor usage:

```yaml
fitness:
  sensor_count_penalty: 0.0
  sensor_length_penalty: 1.0
  sensor_length_scale: 500.0
```

With these values, total sensor length is penalized, while the number of active
sensors is not penalized. This allows simple experiments with the trade-off
between task performance and sensor length.

---

## Difficulty Presets

GapNavigator uses difficulty-specific EvoLib and task configuration files.

| Difficulty | EvoLib config | Task config |
|---|---|---|
| easy | `config_easy.yaml` | `task_easy.yaml` |
| medium | `config_medium.yaml` | `task_medium.yaml` |
| hard | `config_hard.yaml` | `task_hard.yaml` |

Typical changes between presets include:

- obstacle row speed
- obstacle row spacing
- gap width
- number of generations
- network size
- mutation parameters
- reward or sensor penalty settings

Difficulty presets should keep the observation and action interfaces stable.
This makes it possible to change task parameters without changing the surrounding
example code.

---

## Files

| File | Purpose |
|---|---|
| `gap_navigator_play.py` | Manual control with left/right keys or A/D |
| `gap_navigator_rule.py` | Simple sensor-based baseline controller |
| `gap_navigator_train.py` | Evolves sensors and an EvoNet controller |
| `gap_navigator_watch.py` | Loads and visualizes a trained checkpoint |
| `config_easy.yaml` | EvoLib training config for the easy preset |
| `config_medium.yaml` | EvoLib training config for the medium preset |
| `config_hard.yaml` | EvoLib training config for the hard preset |
| `task_easy.yaml` | GapNavigator task config for the easy preset |
| `task_medium.yaml` | GapNavigator task config for the medium preset |
| `task_hard.yaml` | GapNavigator task config for the hard preset |

Package-side support files:

| File | Purpose |
|---|---|
| `evoenv/envs/gap_navigator.py` | Headless environment logic |
| `evoenv/envs/gap_navigator_objects.py` | Player, gap row, and block sprite objects |
| `evoenv/envs/gap_navigator_task.py` | EvoLib task integration, sensor decoding, reward calculation |
| `evoenv/envs/gap_navigator_config.py` | Pydantic task configuration models |
| `evoenv/envs/gap_navigator_defaults.py` | Shared defaults |
| `evoenv/renderers/pygame_gap_navigator.py` | Pygame visualization and GIF recording support |

---

## Run

Manual control with the default difficulty:

```bash
python gap_navigator_play.py
```

Rule-based controller:

```bash
python gap_navigator_rule.py
```

Train evolved sensors and an evolved controller:

```bash
python gap_navigator_train.py
```

Train with debug visualization:

```bash
python gap_navigator_train.py --debug
```

Watch the saved medium individual:

```bash
python gap_navigator_watch.py gap_navigator_medium.pkl
```

Use a specific difficulty:

```bash
python gap_navigator_play.py --difficulty easy
python gap_navigator_train.py --difficulty hard
python gap_navigator_watch.py gap_navigator_hard.pkl
```

---

## Manual Controls

| Key | Action |
|---|---|
| Left arrow / `A` | Steer left |
| Right arrow / `D` | Steer right |
| `R` | Reset episode |
| `ESC` | Quit |

---

## Rule-Based Baseline

The rule-based controller uses only sensor activations and horizontal velocity.
It does not receive the gap center.

It compares obstacle pressure on the left and right sensor groups:

```python
midpoint = len(sensor_values) // 2
left_pressure = sum(sensor_values[:midpoint])
right_pressure = sum(sensor_values[midpoint:])
steering = left_pressure - right_pressure
```

The rule then damps horizontal velocity:

```python
steering -= velocity_x * 0.35
```

This baseline is intentionally simple. It is useful as a sanity check, but it is
not meant to be an optimal policy.

---

## Debug Visualization

During debug training, the current best individual can be visualized and
optionally written as a GIF.

This is useful for checking sensor geometry, steering behavior, collisions, wall
contact, and excessive oscillation.

Debug frames are written to the `frames/` directory when `--debug` is enabled.

---

## Expected Behavior

At the beginning of training, evolved controllers often:

- steer randomly
- collide with early rows
- oscillate or stay near a wall
- use unnecessarily long sensors

After successful training, useful controllers should learn to:

- react before obstacle rows reach the player
- steer toward open gaps
- reduce unnecessary movement
- use shorter sensors when the length penalty makes this advantageous

GapNavigator is more complex than Jumper because both the controller and the ray
sensor layout are evolved. The example should still stay small enough to inspect
visually and modify by hand.
