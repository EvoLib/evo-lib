# 02_jumper – Timing-Based Jumping Example

This example demonstrates how a controller learns to jump over moving obstacles.

Compared with LineFollower, Jumper introduces a different kind of control problem:
the agent must make a discrete timing decision instead of continuously steering.

The environment is intentionally small, easy to visualize, and easy to reason about.
It is designed as an introduction to:

- event timing
- delayed consequences
- binary decisions
- sparse failure conditions
- ray-based obstacle sensing
- evolutionary controller optimization

For a general overview of the interactive environment system, see the main
README in `examples/09_evoenv/`.

---

<p align="center">
  <img src="https://raw.githubusercontent.com/EvoLib/evo-lib/main/examples/09_evoenv/02_jumper/frames/jumper.gif" alt="Jumper example animation" width="512"/>
</p>

---

## Goal

The player should avoid incoming obstacles by jumping at the right time.

The task is solved well when the controller:

- detects an incoming obstacle with the forward ray sensor
- jumps only when the obstacle is close enough
- avoids jumping too early
- avoids unnecessary jumps
- uses the ground state correctly
- stays collision-free for many simulation steps

---

## Sensor Model

Jumper uses the shared EvoEnv sensor architecture.

By default, the environment creates one forward-facing `RaySensor`:

```python
RaySensor(length=250.0, angle=math.pi / 2.0)
```

The sensor starts at the front of the player body and points horizontally toward
incoming obstacles. If the ray intersects an obstacle, the observation receives a
proximity value.

Sensor value semantics:

| Value | Meaning |
|---:|---|
| `0.0` | no obstacle hit within sensor range |
| close to `0.0` | obstacle is near the end of the ray |
| close to `1.0` | obstacle is very close to the player |

The visible debug renderer draws the sensor ray. When an obstacle is hit, the
visible ray is shortened to the hit point.

---

## Observation Space

The environment returns:

```text
[*sensor_values, normalized_player_height, on_ground]
```

With the default one-sensor layout, the observation has three values:

| Index | Value | Meaning |
|---:|---|---|
| 0 | `obstacle_proximity` | Forward ray proximity value in `[0.0, 1.0]` |
| 1 | `normalized_player_height` | Current jump height above the ground in `[0.0, 1.0]` |
| 2 | `on_ground` | `1.0` if the player can jump, else `0.0` |

Typical observations:

| Observation Pattern | Meaning |
|---|---|
| `obstacle_proximity == 0.0` | no obstacle detected by the ray |
| `obstacle_proximity` close to `1.0` | obstacle is close |
| `normalized_player_height > 0.0` | player is currently jumping |
| `on_ground == 1.0` | player can start a jump |
| `on_ground == 0.0` | player is airborne |

If a custom sensor layout is passed to `JumperEnv`, the observation size changes
to:

```text
len(sensors) + 2
```

The final two values are always `normalized_player_height` and `on_ground`.

---

## Action Space

The controller returns two values:

| Index | Value | Meaning |
|---:|---|---|
| 0 | `jump_signal` | Jump trigger signal in `[0.0, 1.0]` |
| 1 | `jump_force` | Jump force control in `[0.0, 1.0]` |

A jump starts only when:

```python
on_ground and jump_signal > 0.5
```

`jump_force` scales the initial jump velocity. The current physics uses:

```python
force_scale = 0.65 + 0.45 * jump_force
```

A typical rule-based jump action is:

```python
[1.0, 0.75]
```

A no-jump action is:

```python
[0.0, 0.0]
```

---

## Rule-Based Baseline

The rule-based controller uses only the forward obstacle proximity and the ground
state:

```python
obstacle_proximity = observation[0]
on_ground = observation[2] >= 0.5

should_jump = on_ground and obstacle_proximity > 0.85
```

This is intentionally simple. It demonstrates the expected timing behavior
without exposing the exact obstacle distance as a direct input.

---

## Reward

The reward is computed inside the environment.

It currently consists of:

| Reward Part | Meaning |
|---|---|
| `alive_reward` | Small reward for surviving a step |
| `pass_reward` | Reward for successfully passing an obstacle |
| `collision_penalty` | Penalty when the player collides with an obstacle |

The episode ends when:

- the maximum step count is reached
- a collision occurs and `terminate_on_collision` is enabled for the selected difficulty

If `terminate_on_collision` is disabled, collisions are penalized but the episode
continues until the maximum step count is reached.

---

## Difficulty Levels

The environment supports three difficulty levels.

Typical changes between difficulties:

- gravity
- jump velocity
- obstacle speed
- obstacle width and height
- obstacle spacing
- collision termination behavior
- reward and penalty values

Examples:

```bash
python jumper_play.py --difficulty easy
python jumper_play.py --difficulty medium
python jumper_play.py --difficulty hard
```

---

## Files

Example-side files:

| File | Purpose |
|---|---|
| `jumper_play.py` | Manual control with the space bar |
| `jumper_rule.py` | Simple sensor-based baseline controller |
| `jumper_train.py` | Evolves an EvoNet controller |
| `jumper_watch.py` | Loads and visualizes a trained checkpoint |

Package-side support files:

| File | Purpose |
|---|---|
| `evoenv/envs/jumper.py` | Headless environment logic and ray-based sensing |
| `evoenv/envs/jumper_objects.py` | Player and obstacle simulation objects |
| `evoenv/envs/jumper_settings.py` | Difficulty presets |
| `evoenv/envs/jumper_task.py` | EvoLib task integration |
| `evoenv/envs/jumper_defaults.py` | Shared defaults |
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

Watch the best saved individual:

```bash
python jumper_watch.py jumper_medium.pkl
```

---

## Debug Visualization

During debug training, the current best individual can be visualized and exported
as an animated GIF.

This is useful for:

- inspecting jump timing
- checking whether the ray sensor detects obstacles as expected
- detecting reward problems
- comparing evolved behavior with the rule-based baseline
- observing whether the controller jumps too early or too late

---

## Expected Behavior

At the beginning of training, evolved controllers often:

- collide quickly
- jump too early
- jump continuously
- ignore the ground state
- react too late to the sensor signal

After several generations, useful controllers should learn to:

- wait while no obstacle is detected
- jump when the sensor proximity becomes high
- avoid repeated useless jumps
- use the ground state to avoid wasted jump signals while airborne
- survive longer episodes

The rule-based controller is intentionally simple and strong enough to show what
a plausible controller behavior looks like.

---

## Didactic Value

Jumper introduces concepts that are not visible in pure steering tasks:

- timing
- delayed action effects
- discrete decisions
- failure by collision
- action thresholds
- ray-based sensing
- observation-dependent control

Compared with LineFollower, this task is less about continuous correction and
more about choosing the right moment for an action.
