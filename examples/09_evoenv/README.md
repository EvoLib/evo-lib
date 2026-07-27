# EvoEnv Examples

This directory contains small Pygame environments for experimenting with EvoLib
controllers.

Each example follows the same workflow:

```text
play -> rule -> train -> watch
```

| Script | Purpose |
|---|---|
| `*_play.py` | Explore the environment manually |
| `*_rule.py` | Run a simple hand-written baseline |
| `*_train.py` | Evolve an EvoLib controller |
| `*_watch.py` | Visualize a saved checkpoint |

The package architecture and conventions are documented in
[`evoenv/README.md`](../../evoenv/README.md).

---

## 01 — Line Follower

<p align="center">
  <img src="https://raw.githubusercontent.com/EvoLib/evo-lib/main/examples/09_evoenv/01_line_follower/linefollower.gif" alt="LineFollower sample" width="512"/>
</p>

A small agent follows a path using local point sensors and continuous steering.
The example introduces the basic relationship between sensor observations,
controller output, and movement. Difficulty presets vary the path and sensor
requirements without changing the overall example structure.

[Open Line Follower](01_line_follower/)

---

## 02 — Jumper

<p align="center">
  <img src="https://raw.githubusercontent.com/EvoLib/evo-lib/main/examples/09_evoenv/02_jumper/jumper.gif" alt="Jumper example animation" width="512"/>
</p>

The controller moves forward and must trigger jumps at suitable times to avoid
obstacles. The example focuses on action timing, compact observations, and
episodic success or failure. It is useful for inspecting how small changes in
sensor input affect discrete behavior.

[Open Jumper](02_jumper/)

---

## 03 — Gap Navigator

<p align="center">
  <img src="https://raw.githubusercontent.com/EvoLib/evo-lib/main/examples/09_evoenv/03_gap_navigator/gap_navigator.gif" alt="GapNavigator sample" width="512"/>
</p>

An agent moves through successive obstacle rows and must steer toward open gaps.
The example combines directional sensing, collision avoidance, and configurable
sensor layouts. It is suited to experiments with small structural changes in a
controller or its available sensors.

[Open Gap Navigator](03_gap_navigator/)

---

## 04 — Collector

<p align="center">
  <img src="https://raw.githubusercontent.com/EvoLib/evo-lib/main/examples/09_evoenv/04_collector/collector.gif" alt="Collector sample" width="512"/>
</p>

An agent moves freely through a 2D arena, follows food targets, and avoids walls
and rectangular obstacles. The example combines continuous steering and
throttle control with target observations, local ray sensors, and configurable
reward shaping.

[Open Collector](04_collector/)
