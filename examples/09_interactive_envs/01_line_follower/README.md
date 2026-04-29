# LineFollower – Example

This example demonstrates how a robot learns to follow a line using evolution.

For a full explanation of the learning pipeline and architecture, see the main
README in `examples/09_interactive_envs/`.

---

## Description

A robot follows a line using two binary sensors.

- Observation: [left_sensor, right_sensor]
- Action: [turn] in [-1, 1]

The goal is to stay on the line and move forward.

---

## Scripts

### play.py
Manual control to understand the environment.

### rule.py
Simple rule-based controller (baseline).

### train.py
Evolve a neural network controller.

### watch.py
Visualize the best trained individual.

---

## Notes

- Minimal state space (2 sensors)
- Fast training
- Focus on understanding, not performance
