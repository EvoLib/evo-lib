# 02_jumper – Timing-Based Jumping Example

This example demonstrates how a controller learns to jump over moving obstacles.

Compared with LineFollower, Jumper introduces a different kind of control problem:
the agent must make a discrete timing decision instead of continuously steering.

The environment is intentionally small, visual, and easy to reason about.

It is designed as an introduction to:

- event timing
- delayed consequences
- binary decisions
- sparse failure conditions
- evolutionary controller optimization

For a general overview of the interactive environment system, see the main
README in `examples/09_interactive_envs/`.

---

<p align="center">
  <img src="https://github.com/EvoLib/evo-lib/blob/feature/envs-pygame/examples/09_interactive_envs/02_jumper/frames/jumper.gif?raw=true" alt="Sample Plot" width="512"/>
</p>

---

# Goal

The player should avoid obstacles by jumping at the right time.

The task is solved well when the controller:

- jumps shortly before an obstacle reaches the player
- avoids jumping too early
- avoids unnecessary jumps
- uses the ground state correctly
- survives for many simulation steps

---

# Observation Space

The environment returns three values:

| Index | Value | Meaning |
|---:|---|---|
| 0 | `normalized_distance` | Distance from player to obstacle, normalized to `[0.0, 1.0]` |
| 1 | `normalized_player_height` | Current jump height, normalized to `[0.0, 1.0]` |
| 2 | `on_ground` | `1.0` if the player can jump, else `0.0` |

Typical observations:

| Observation Pattern | Meaning |
|---|---|
| `normalized_distance` close to `1.0` | obstacle is far away |
| `normalized_distance` close to `0.0` | obstacle is close |
| `normalized_player_height > 0.0` | player is currently jumping |
| `on_ground == 1.0` | player can start a jump |
| `on_ground == 0.0` | player is airborne |

---

# Action Space

The controller returns two values:

| Index | Value | Meaning |
|---:|---|---|
| 0 | `jump_signal` | Jump if value is high enough |
| 1 | `jump_strength` | Jump strength or jump intensity |

A typical rule-based action is:

```python
[1.0, 0.75]
```

for jumping, and:

```python
[0.0, 0.0]
```

for no jump.

---

# Reward

The reward function encourages:

- surviving longer
- avoiding collisions
- jumping at useful times
- avoiding unnecessary behavior

The episode usually ends when:

- the player collides with an obstacle
- the maximum step count is reached

---

# Difficulty Levels

The environment supports multiple difficulty levels.

Typical changes between difficulties:

- obstacle speed
- obstacle spacing
- episode length
- timing tolerance

Examples:

```bash
python jumper_play.py --difficulty easy
python jumper_play.py --difficulty medium
python jumper_play.py --difficulty hard
```

---

# Files

| File | Purpose |
|---|---|
| `jumper_play.py` | Manual control with the space bar |
| `jumper_rule.py` | Simple rule-based baseline controller |
| `jumper_train.py` | Evolves an EvoNet controller |
| `jumper_watch.py` | Loads and visualizes a trained checkpoint |

Package-side support files:

| File | Purpose |
|---|---|
| `evolib_envs/envs/jumper.py` | Headless environment logic |
| `evolib_envs/envs/jumper_task.py` | EvoLib task integration |
| `evolib_envs/envs/jumper_defaults.py` | Shared defaults |
| `evolib_envs/renderers/pygame_jumper.py` | Pygame visualization |

---

# Run

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

# Debug Visualization

During debug training, the current best individual can be visualized.

This is useful for:

- inspecting jump timing
- detecting reward problems
- comparing evolved behavior with the rule-based baseline
- observing whether the controller jumps too early or too late

---

# Expected Behavior

At the beginning of training, evolved controllers often:

- collide quickly
- jump too early
- jump continuously
- ignore the ground state

After several generations, useful controllers should learn to:

- wait while the obstacle is far away
- jump only when the obstacle is close
- avoid repeated useless jumps
- survive longer episodes

The rule-based controller is intentionally simple and strong enough to show what
a working policy looks like.

---

Jumper introduces concepts that are not visible in pure steering tasks:

- timing
- delayed action effects
- discrete decisions
- failure by collision
- action thresholds
- observation-dependent control

Compared with LineFollower, this task is less about continuous correction and
more about choosing the right moment for an action.
