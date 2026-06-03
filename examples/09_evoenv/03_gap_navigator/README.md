# 03_gap_navigator – Evolvable Sensor Navigation Example

This example demonstrates how a controller learns to steer through falling obstacle rows with free gaps.

Compared with Jumper, GapNavigator introduces a more open continuous-control problem:
the agent does not receive the gap position directly. Instead, it receives ray sensor values and must learn how to steer based on obstacle proximity.

It is designed as an introduction to:

- continuous horizontal steering
- evolved sensor layouts
- proximity-based obstacle avoidance
- reward shaping for navigation tasks
- separating task state from rendered debug information
- evolutionary controller optimization with multiple parameter modules

For a general overview of the interactive environment system, see the main
README in `examples/09_evoenv/`.

---

<p align="center">
  <img src="https://raw.githubusercontent.com/EvoLib/evo-lib/main/examples/09_evoenv/03_gap_navigator/frames/gap_navigator.gif" alt="GapNavigator sample" width="512"/>
</p>

---

# Goal

The player should move horizontally and pass through the gaps in falling obstacle rows.

The task is solved well when the controller:

- reacts to approaching obstacle blocks
- moves toward free gaps early enough
- avoids oversteering
- avoids staying near the screen edges
- passes many obstacle rows without collisions
- uses a compact and useful sensor layout if sensor penalties are enabled

---

# Task Structure

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

The task converts this vector into `RaySensor` objects. Very short sensors can be represented as zero-length inactive sensors, depending on `min_active_length`.

---

# Observation Space

The environment returns `max_sensors + 2` values.

With the default `max_sensors: 6`, the observation size is `8`.

| Index | Value | Meaning |
|---:|---|---|
| `0..max_sensors-1` | `sensor_value` | Proximity activation for each ray sensor |
| `max_sensors` | `normalized_x` | Horizontal player position normalized to `[0.0, 1.0]` |
| `max_sensors + 1` | `normalized_velocity_x` | Horizontal velocity normalized by player speed |

Typical observations:

| Observation Pattern | Meaning |
|---|---|
| sensor value close to `0.0` | no nearby obstacle hit on that ray |
| sensor value close to `1.0` | obstacle hit very close to the player |
| `normalized_x` close to `0.0` | player is near the left edge |
| `normalized_x` close to `1.0` | player is near the right edge |
| `normalized_velocity_x < 0.0` | player is moving left |
| `normalized_velocity_x > 0.0` | player is moving right |

The controller does not receive the gap center directly. This is intentional. The controller must infer useful steering behavior from sensor activations and self-state.

---

# Action Space

The controller returns one value:

| Index | Value | Meaning |
|---:|---|---|
| 0 | `steering` | Horizontal steering clipped to `[-1.0, 1.0]` |

A typical action is:

```python
[-1.0]
```

for full left steering,

```python
[1.0]
```

for full right steering, and:

```python
[0.0]
```

for no horizontal movement.

---

# Reward

The task-level reward encourages:

- alignment with the next relevant gap
- avoiding obstacle collisions
- avoiding unnecessary movement
- avoiding positions near screen edges
- optionally passing rows directly when `terminate_on_collision` is enabled

The default task configuration uses shaped feedback instead of only rewarding completed passes:

```yaml
reward:
  pass_reward: 0.0
  gap_alignment_reward: 0.040
  movement_penalty: 0.014
  collision_penalty: 6.5
  near_wall_penalty: 0.040
```

The fitness function can additionally penalize sensor usage:

```yaml
fitness:
  sensor_count_penalty: 0.0
  sensor_length_penalty: 1.0
  sensor_length_scale: 500.0
```

This makes the example useful for studying the trade-off between controller performance and sensor complexity.

---

# Difficulty Levels

The environment supports multiple difficulty levels.

Typical changes between difficulties:

- obstacle row speed
- obstacle row spacing
- gap width
- number of generations
- network size
- mutation strength

Examples:

```bash
python gap_navigator_play.py --difficulty easy
python gap_navigator_play.py --difficulty medium
python gap_navigator_play.py --difficulty hard
```

---

# Files

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
| `evoenv/envs/gap_navigator_objects.py` | Player, obstacle row, and block sprite objects |
| `evoenv/envs/gap_navigator_task.py` | EvoLib task integration, sensor decoding, reward calculation |
| `evoenv/envs/gap_navigator_config.py` | Pydantic task configuration models |
| `evoenv/envs/gap_navigator_defaults.py` | Shared defaults |
| `evoenv/renderers/pygame_gap_navigator.py` | Pygame visualization and GIF recording support |

---

# Run

Manual control:

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

Watch the best saved individual:

```bash
python gap_navigator_watch.py gap_navigator_medium.pkl
```

Use a specific difficulty:

```bash
python gap_navigator_train.py --difficulty hard
python gap_navigator_watch.py gap_navigator_hard.pkl
```

---

# Manual Controls

| Key | Action |
|---|---|
| Left arrow / `A` | Steer left |
| Right arrow / `D` | Steer right |
| `R` | Reset episode |
| `ESC` | Quit |

---

# Rule-Based Baseline

The rule-based controller uses only sensor activations and horizontal velocity.

It compares obstacle pressure on the left and right sensor groups:

```python
midpoint = len(sensor_values) // 2
left_pressure = sum(sensor_values[:midpoint])
right_pressure = sum(sensor_values[midpoint:])
steering = left_pressure - right_pressure
```

This baseline is intentionally simple. It does not know the gap center. It only steers away from the side with stronger obstacle activation and damps horizontal velocity.

---

# Debug Visualization

During debug training, the current best individual can be visualized and optionally written as a GIF.

This is useful for:

- inspecting evolved sensor length and angle patterns
- detecting reward problems
- checking whether collisions are caused by late steering or bad sensor geometry
- comparing evolved behavior with the rule-based baseline
- observing whether the agent exploits wall contact, oscillation, or oversteering

Debug frames are written to the `frames/` directory by `train.py` when `--debug` is enabled.

---

# Expected Behavior

At the beginning of training, evolved controllers often:

- steer randomly
- collide with the first few obstacle rows
- oscillate horizontally
- overuse long sensors
- ignore useful sensor activations
- get stuck near a wall

After several generations, useful controllers should learn to:

- react before obstacle rows reach the player
- keep enough distance from solid blocks
- steer toward open gaps
- reduce unnecessary movement
- survive longer episodes
- use shorter or fewer sensors when sensor penalties make this advantageous

The rule-based controller is useful as a sanity check, but it is not meant to be an optimal policy.

---

GapNavigator introduces concepts that are not visible in simpler timing tasks:

- evolved perception
- sensor geometry as part of the search space
- continuous steering from sparse proximity signals
- reward shaping for navigation
- behavioral side effects of sensor penalties
- separation between hidden task diagnostics and actual observations

Compared with Jumper, this task is less about choosing the right moment for a discrete action and more about building a compact perception-control loop.
