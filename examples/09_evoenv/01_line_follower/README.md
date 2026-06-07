# 01_line_follower – Sensor-Based Steering Example

This example demonstrates how an evolved controller can steer a robot along a
line using simple binary sensor feedback.

LineFollower is intentionally small. It introduces the basic EvoEnv control loop
before the later examples add jump timing, evolved sensors, or more complex
obstacle layouts.

This example focuses on:

- binary line sensors
- steering actions
- reward from line contact and forward progress
- comparing rule-based and evolved controllers
- optional difficulty presets with unchanged observations and actions

For a general overview of the Pygame-based EvoEnv examples, see the main README
in `examples/09_evoenv/`.

---

<p align="center">
  <img src="https://raw.githubusercontent.com/EvoLib/evo-lib/main/examples/09_evoenv/01_line_follower/frames/linefollower.gif" alt="LineFollower sample" width="512"/>
</p>

---

## Goal

The robot should stay close to the line while continuously moving forward.

A useful controller should:

- correct steering smoothly
- avoid losing the track
- recover from small deviations
- remain on the line for many simulation steps

---

## Task Configuration

LineFollower uses difficulty-specific task configuration files:

| Difficulty | EvoLib config | Task config |
|---|---|---|
| easy | `config_easy.yaml` | `task_easy.yaml` |
| medium | `config_medium.yaml` | `task_medium.yaml` |
| hard | `config_hard.yaml` | `task_hard.yaml` |

The EvoLib config controls the evolutionary setup, network size, and mutation
parameters. The task config controls the environment and reward parameters.

The observation and action spaces stay unchanged across all difficulty presets.
This makes it possible to compare the same controller interface under different
line shapes, speeds, and tolerances.

---

## Observation Space

The environment returns two sensor values.

| Index | Value | Meaning |
|---:|---|---|
| 0 | `left_sensor` | Line contact value of the left sensor |
| 1 | `right_sensor` | Line contact value of the right sensor |

The sensors are binary and detect whether they overlap with the line.

Typical observations:

| Observation | Meaning |
|---|---|
| `[1.0, 1.0]` | both sensors detect the line |
| `[1.0, 0.0]` | robot is drifting right |
| `[0.0, 1.0]` | robot is drifting left |
| `[0.0, 0.0]` | robot lost the line |

---

## Action Space

The controller returns one steering value.

| Index | Value | Meaning |
|---:|---|---|
| 0 | `turn` | Steering value in range `[-1.0, 1.0]` |

Typical behavior:

| Value | Meaning |
|---|---|
| `-1.0` | strong left turn |
| `0.0` | drive straight |
| `1.0` | strong right turn |

---

## Controller Network

The easy and medium training configs use a very small EvoNet controller:

```yaml
modules:
  brain:
    type: evonet
    dim: [2, 1, 1]
    activation: [linear, tanh, tanh]
```

This matches the environment interface:

| Layer | Size | Meaning |
|---|---:|---|
| input | 2 | `left_sensor`, `right_sensor` |
| hidden | 1 | minimal nonlinear controller layer |
| output | 1 | steering action |

The hard training config may use a larger controller and population. It should be
understood as a matching training profile for the harder task preset, not as a
change to the observation or action interface.

---

## Reward

The reward is computed directly by the environment.

The default task configs use:

```yaml
reward:
  progress_reward_scale: 0.25
  missed_line_penalty: 0.25
```

The reward behavior is:

```text
if robot touches the line:
    reward += forward_progress * progress_reward_scale
else:
    reward -= missed_line_penalty
```

The episode ends when:

- the maximum step count is reached
- the robot misses the line for too many consecutive steps
- the robot reaches the end of the screen
- the robot leaves the screen bounds

EvoLib uses minimization, so training converts accumulated reward to fitness via:

```python
indiv.fitness = -reward
```

---

## Difficulty Presets

Difficulty presets change task parameters, not the controller interface.

Typical changes between difficulties:

- line curvature
- line width
- movement speed
- steering strength
- tolerated missed-line steps

Examples:

```bash
python line_follower_play.py --difficulty easy
python line_follower_play.py --difficulty medium
python line_follower_play.py --difficulty hard
```

---

## Files

| File | Purpose |
|---|---|
| `config_easy.yaml` | EvoLib training config for the easy preset |
| `config_medium.yaml` | EvoLib training config for the medium preset |
| `config_hard.yaml` | EvoLib training config for the hard preset |
| `task_easy.yaml` | LineFollower task config for the easy preset |
| `task_medium.yaml` | LineFollower task config for the medium preset |
| `task_hard.yaml` | LineFollower task config for the hard preset |
| `line_follower_play.py` | Manual steering with keyboard input |
| `line_follower_rule.py` | Simple rule-based steering controller |
| `line_follower_train.py` | Evolves an EvoNet controller |
| `line_follower_watch.py` | Loads and visualizes a trained checkpoint |

Package-side support files:

| File | Purpose |
|---|---|
| `evoenv/envs/line_follower.py` | Headless environment logic |
| `evoenv/envs/line_follower_config.py` | Pydantic task configuration models |
| `evoenv/envs/line_follower_task.py` | EvoLib task integration |
| `evoenv/envs/line_follower_objects.py` | Robot and sensor objects |
| `evoenv/envs/line_follower_defaults.py` | Shared runtime defaults |
| `evoenv/renderers/pygame_line_follower.py` | Pygame visualization |

---

## Run

Manual control:

```bash
python line_follower_play.py
```

Rule-based controller:

```bash
python line_follower_rule.py
```

Train an evolved controller:

```bash
python line_follower_train.py
```

Train with debug visualization:

```bash
python line_follower_train.py --debug
```

Watch the best saved individual:

```bash
python line_follower_watch.py line_follower_medium.pkl
```

Use a specific difficulty:

```bash
python line_follower_train.py --difficulty hard
python line_follower_watch.py line_follower_hard.pkl
```

---

## Manual Controls

| Key | Action |
|---|---|
| Left arrow | Steer left |
| Right arrow | Steer right |
| `R` | Reset episode |
| `ESC` | Quit |

---

## Rule-Based Baseline

The rule controller uses the two sensor values directly:

```python
left_sensor, right_sensor = observation
turn = right_sensor - left_sensor
```

The rule steers toward the side whose sensor has lost the line. It is intended as
a simple sanity check, not as an optimized controller.

---

## Debug Visualization

During debug training, the current best individual can be visualized.

This is useful for checking steering behavior, line contact, reward shaping, and
whether the robot loses the line early.

Debug frames are written to the `frames/` directory when `--debug` is enabled.

---

## Expected Behavior

At the beginning of training, evolved controllers often:

- spin in circles
- oversteer
- lose the line quickly

After successful training, useful controllers should learn to:

- keep the line centered between both sensors
- correct small deviations
- reduce line misses
- progress farther along the line
