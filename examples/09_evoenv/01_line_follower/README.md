# 01_line_follower – Sensor-Based Steering Example

This example demonstrates how a controller learns to steer a robot along a line
using simple sensor feedback.

The environment is intentionally small and easy to understand.

It is designed as an introduction to:

- observations
- actions
- rewards
- controller behavior
- evolutionary learning

For a general overview of the interactive environment system, see the main
README in `examples/09_interactive_envs/`.

---

<p align="center">
  <img src="https://raw.githubusercontent.com/EvoLib/evo-lib/main/examples/09_evoenv/01_line_follower/frames/linefollower.gif" alt="Sample Plot" width="512"/>
</p>

---

# Goal

The robot should stay close to the line while continuously moving forward.

The task is solved well when the robot:

- corrects steering smoothly
- avoids losing the track
- survives for many simulation steps

---

# Observation Space

The environment returns two sensor values:

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

# Action Space

The controller returns one steering value:

| Index | Value | Meaning |
|---:|---|---|
| 0 | `turn` | Steering value in range `[-1.0, 1.0]` |

Typical interpretation:

| Value | Meaning |
|---|---|
| `-1.0` | strong left turn |
| `0.0` | drive straight |
| `1.0` | strong right turn |

---

# Reward

The reward function encourages:

- staying near the line
- stable steering behavior
- long survival time

The episode ends when:

- the robot loses the line for too long
- the maximum step count is reached

---

# Difficulty Levels

The environment supports multiple difficulty levels.

Typical changes between difficulties:

- line curvature
- line width
- movement speed

Examples:

```bash
python line_follower_play.py --difficulty easy
python line_follower_play.py --difficulty medium
python line_follower_play.py --difficulty hard
```

---

# Files

| File | Purpose |
|---|---|
| `line_follower_play.py` | Manual steering with keyboard input |
| `line_follower_rule.py` | Simple rule-based steering controller |
| `line_follower_train.py` | Evolves an EvoNet controller |
| `line_follower_watch.py` | Loads and visualizes a trained checkpoint |

Package-side support files:

| File | Purpose |
|---|---|
| `evolib_envs/envs/line_follower.py` | Headless environment logic |
| `evolib_envs/envs/line_follower_task.py` | EvoLib task integration |
| `evolib_envs/envs/line_follower_defaults.py` | Shared defaults |
| `evolib_envs/renderers/pygame_line_follower.py` | Pygame visualization |

---

# Run

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
python watch.py
```

---

# Debug Visualization

During debug training, the current best individual can be visualized.

This is useful for:

- inspecting steering behavior
- debugging reward shaping
- observing learning progress

Training also allowes GIF animation export.

Example output:

```text
frames/gen_025.gif
```

---

# Expected Behavior

At the beginning of training, evolved controllers often:

- spin in circles
- oversteer
- lose the line quickly

After several generations, useful controllers should learn to:

- keep the line centered
- perform smooth corrections
- recover from small deviations

