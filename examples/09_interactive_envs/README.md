# Interactive Environments (EvoLib)

Interactive Environments provide small, visual, and reproducible learning environments
for EvoLib.

The goal is not to build high-performance game environments, the goal is to make:

- evolutionary algorithms
- neural networks
- controller behavior
- emergent strategies

observable and understandable.

These environments are designed primarily for:

- experimentation
- demonstrations
- step-by-step learning

---

# Design Goals

The project focuses on:

- small and understandable environments
- immediate visual feedback
- short iteration cycles
- deterministic and reproducible runs
- clean separation between simulation and rendering
- reusable evaluation pipelines

---

# Learning Pipeline

Every environment follows the same learning pipeline:

```text
play -> rule -> train -> watch
```

This progression is consistent across all examples.

---

## 1. play.py

The user controls the agent manually.

Goal:

- understand the environment
- understand controls
- develop intuition for the task

---

## 2. rule.py

A simple rule-based controller solves the task heuristically.

Goal:

- demonstrate that the task is solvable
- provide a baseline behavior
- show the observation/action relationship

---

## 3. train.py

EvoLib evolves a neural network controller.

Goal:

- observe learning progress
- inspect evolutionary improvement
- experiment with parameters and mutation behavior

Training is usually headless for performance reasons.

Optional debug visualization can be enabled during training.

---

## 4. watch.py

Loads a saved checkpoint and visualizes the best evolved individual.

Goal:

- observe emergent behavior
- compare evolved strategies
- inspect training results

---

# Core Architecture

The environments are separated into reusable components.

---

## Env

The environment contains the simulation logic.

```python
class Env:
    def reset(self, seed=None): ...
    def step(self, action): ...
```

Responsibilities:

- simulation state
- observations
- rewards
- episode termination

The environment itself is renderer-independent.

---

## Controller

Controllers map observations to actions.

```python
class Controller:
    def act(self, observation): ...
```

Typical controller types:

- manual controllers
- rule-based controllers
- EvoNet controllers

---

## Task

Tasks connect EvoLib individuals with environments.

Responsibilities:

- evaluate individuals
- run episodes
- compute fitness
- provide visualization helpers

Tasks are the primary integration layer between EvoLib and the environments.

---

## Renderer

Renderers visualize environment state using Pygame.

Responsibilities:

- drawing
- overlays
- sensor visualization
- debug output

Rendering is separated from simulation.

This allows:

- headless training
- reproducible evaluation
- parallel execution

---

## Checkpoints

Training results are stored as checkpoints.

Typical checkpoint contents:

- evolved individual
- environment name
- difficulty
- random seed

This allows:

- reproducible demonstrations
- loading trained controllers
- separating training from visualization

---

# Difficulty System

Environments can provide multiple difficulty levels:

- easy
- medium
- hard

Difficulty affects environment parameters while keeping the same observation
and action interfaces.

This allows progressive learning without changing the surrounding code.

---

# Debug Visualization

Training can optionally visualize the current best individual.

Typical use cases:

- debugging reward functions
- inspecting controller behavior
- observing learning progress

Example:

```bash
python jumper_train.py --debug
```

---

# GIF Export

Some environments support GIF export during training.

This is useful for:

- comparing generations
- documentation
- regression inspection

Example output:

```text
frames/gen_025.gif
```

---

# Current Environments

| Environment | Focus |
|---|---|
| LineFollower | steering and sensor feedback |
| Jumper | timing and event-based decisions |

---

# Didactic Philosophy

The environments intentionally avoid unnecessary complexity.

The focus is on:

- understanding behavior
- understanding observations
- understanding rewards
- understanding evolution

Small environments are easier to reason about and easier to modify.

This makes them suitable for:

- classrooms
- workshops
- self-study
- algorithm experiments

---

# Planned Environments

Future environments:

- FlappyBird
- Pong
- LunarLander
- Spaceship
- Predator-Prey
- Colony
- Multi-agent environments

---

# Technical Direction

Planned future improvements include:

- BatchEnv integration
- parallel evaluation
- Ray-based scaling
- improved logging and tracking

---

# Requirements

Minimum requirements:

- Python 3.10+
- pygame

Install pygame:

```bash
pip install pygame
```
