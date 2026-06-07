# 02_jumper – Sensor-Based Jump Timing and Strength Control

This example demonstrates how an evolved controller can learn to jump over
moving obstacles.

Compared with LineFollower, Jumper introduces timing decisions and multi-frame
movement effects: the controller must trigger a jump before the obstacle reaches
the player, because the resulting jump trajectory unfolds over several simulation
frames.

This example focuses on:

- sensor-based jump timing
- action thresholds
- jump strength control
- collision and jump-strength costs

For a general overview of the Pygame-based EvoEnv examples, see the main README
in `examples/09_evoenv/`.

---

<p align="center">
  <img src="https://raw.githubusercontent.com/EvoLib/evo-lib/main/examples/09_evoenv/02_jumper/frames/jumper.gif" alt="Jumper example animation" width="512"/>
</p>

---

## Goal

The player should avoid incoming obstacles by jumping at the right time and with
sufficient jump strength.

The task is solved well when the controller:

- reacts when the obstacle sensor becomes active
- jumps before the obstacle reaches the player
- uses higher jump strength for higher obstacles
- avoids unnecessarily strong jumps
- reduces collision frames over the episode

Jumper is not a pure binary timing task. The controller has to learn:

```text
When should the player jump?
How strong should the jump be?
```

---

## Task Configuration

Jumper uses a single task configuration file:

```text
task.yaml
```

This file contains the environment, reward, and sensor parameters. The EvoLib
controller configuration remains in `config.yaml`.

Jumper intentionally has no `easy`, `medium`, or `hard` difficulty presets. It is
kept as one standard compact example.

---

## Observation Space

The environment returns two values.

| Index | Value | Meaning |
|---:|---|---|
| 0 | `sensor_value` | Generic ray-sensor value in `[0.0, 1.0]`. Usually `0.0` means inactive or no obstacle hit; values close to `1.0` mean a strong nearby hit. |
| 1 | `normalized_obstacle_height` | Height of the nearest obstacle in front of the player, normalized to `[0.0, 1.0]` using the configured obstacle height range. |

Typical interpretation:

| Observation Pattern | Meaning |
|---|---|
| `sensor_value == 0.0` | no obstacle is currently detected |
| `sensor_value` close to `1.0` | obstacle is very close |
| `normalized_obstacle_height` close to `0.0` | low obstacle |
| `normalized_obstacle_height` close to `1.0` | high obstacle |

The default EvoNet input dimension is therefore `2`.

---

## Action Space

The controller returns two values.

| Index | Value | Meaning |
|---:|---|---|
| 0 | `jump_signal` | Jump trigger. Values greater than `0.5` request a jump. |
| 1 | `jump_strength` | Jump impulse strength in `[0.0, 1.0]`. |

A typical rule-based jump action is:

```python
[1.0, 0.75]
```

A no-jump action is:

```python
[0.0, 0.0]
```

A jump is only triggered when the player is on the ground, `jump_signal > 0.5`,
and `jump_strength > 0.0`.

---

## Controller Network

The default training config uses a small EvoNet controller:

```yaml
modules:
  brain:
    type: evonet
    dim: [2, 4, 2]
    activation: [linear, tanh, sigmoid]
```

This matches the environment interface:

| Layer | Size | Meaning |
|---|---:|---|
| input | 2 | `sensor_value`, `normalized_obstacle_height` |
| hidden | 4 | small nonlinear controller layer |
| output | 2 | `jump_signal`, `jump_strength` |

The sigmoid output layer is useful because both action values are interpreted in
`[0.0, 1.0]`.

---

## Reward

The default reward is cost-focused.

The controller is penalized for:

- colliding with obstacles
- using jump strength when a jump is actually triggered

The default `task.yaml` uses:

| Setting | Value | Meaning |
|---|---:|---|
| `collision_penalty` | `10.0` | penalty for each collision frame |
| `jump_strength_penalty` | `5.0` | penalty multiplier for actual jump strength |
| `pass_reward` | `0.0` | no explicit reward for passing an obstacle |
| `alive_reward` | `0.0` | no passive survival reward |
| `terminate_on_collision` | `false` | collisions are penalized but do not end the episode by default |

The effective reward per step is conceptually:

```text
reward = 0

if collision:
    reward -= collision_penalty

if did_jump:
    reward -= jump_strength**2 * jump_strength_penalty
```

Training minimizes collision frames and unnecessary jump strength. EvoLib uses
minimization, so training converts accumulated reward to fitness via:

```python
indiv.fitness = -reward
```

---

## Default Task Settings

The current standard `task.yaml` uses:

| Setting | Value |
|---|---:|
| `width` | `800` |
| `height` | `450` |
| `max_steps` | `1500` |
| `gravity` | `0.70` |
| `jump_velocity` | `15.5` |
| `obstacle_speed` | `5.0` |
| `obstacle_width` | `35` |
| `min_obstacle_height` | `25` |
| `max_obstacle_height` | `150` |
| `min_spawn_gap` | `250` |
| `max_spawn_gap` | `380` |
| sensor length | `250.0` |
| sensor angle | `1.57079632679` |

Obstacle heights vary between `min_obstacle_height` and `max_obstacle_height`.
This keeps `normalized_obstacle_height` meaningful and gives the controller a
reason to adapt jump strength.

---

## Files

| File | Purpose |
|---|---|
| `config.yaml` | EvoLib training config for the Jumper controller |
| `task.yaml` | Jumper environment, reward, and sensor config |
| `jumper_play.py` | Manual control with the space bar |
| `jumper_rule.py` | Simple sensor-based rule controller |
| `jumper_train.py` | Evolves an EvoNet controller |
| `jumper_watch.py` | Loads and visualizes a trained checkpoint |

Package-side support files:

| File | Purpose |
|---|---|
| `evoenv/envs/jumper.py` | Headless environment logic |
| `evoenv/envs/jumper_objects.py` | Player and obstacle sprites |
| `evoenv/envs/jumper_config.py` | Pydantic task configuration models |
| `evoenv/envs/jumper_defaults.py` | Shared size and debug defaults |
| `evoenv/envs/jumper_task.py` | EvoLib task integration |
| `evoenv/renderers/pygame_jumper.py` | Pygame visualization |

---

## Run

Manual control:

```bash
python jumper_play.py
```

Rule-based controller:

```bash
python jumper_rule.py
```

Train an evolved controller:

```bash
python jumper_train.py
```

Train with debug visualization:

```bash
python jumper_train.py --debug
```

Watch the saved individual:

```bash
python jumper_watch.py jumper.pkl
```

---

## Debug Visualization

During debug training, the current best individual can be visualized.

This is useful for checking jump timing, sensor activation, collision behavior,
and excessive jump strength.

Debug frames are written to the `frames/` directory by default.

---

## Rule-Based Baseline

The rule controller uses the two observation values directly:

```python
sensor_value = observation[0]
normalized_obstacle_height = observation[1]

should_jump = 0.58 <= sensor_value <= 0.92
jump_strength = 0.65 + 0.35 * normalized_obstacle_height
```

This baseline is intentionally simple. It demonstrates the intended control
structure without being an optimized solution:

```text
sensor active enough -> jump
higher obstacle      -> stronger jump
```

---

## Expected Behavior

At the beginning of training, evolved controllers often:

- ignore the sensor
- jump too early or too late
- use excessive jump strength

After successful training, useful controllers should learn to:

- jump when the sensor reaches a useful range
- adapt jump strength to obstacle height
- reduce collision frames and unnecessary jump strength
