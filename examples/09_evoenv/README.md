# EvoEnv

EvoEnv contains small Pygame-based example environments for EvoLib.

The goal is not to build high-performance game environments. The goal is to make
small controller tasks visible and easy to inspect:

- observations
- actions
- rewards
- controller behavior

These examples are mainly intended for:

- experimenting with EvoLib controllers
- testing small environment ideas
- comparing rule-based and evolved behavior

---

## Design Goals

The examples focus on:

- small and understandable environments
- immediate visual feedback
- short iteration cycles
- reproducible evaluation
- clear separation between simulation and rendering
- reusable evaluation code

The environments stay simple enough to understand and modify without
turning into full game projects.

---

## Example Workflow

The current examples use a simple workflow:

```text
play -> rule -> train -> watch
```

This keeps each environment easy to inspect manually before training an evolved
controller.

---

### 1. `*_play.py`

The user controls the agent manually.

Purpose:

- understand the environment
- understand controls
- develop intuition for the task

---

### 2. `*_rule.py`

A simple rule-based controller demonstrates a hand-written baseline.

Purpose:

- check that the task is solvable
- show the observation/action relationship
- provide behavior for comparison

---

### 3. `*_train.py`

EvoLib evolves a neural network controller.

Purpose:

- train an EvoNet controller
- inspect evolutionary improvement
- experiment with mutation and configuration parameters

Training is typically performed headless for performance reasons. Optional debug
visualization can be enabled during training.

---

### 4. `*_watch.py`

Loads a saved checkpoint and visualizes the trained individual.

Purpose:

- inspect the trained controller
- compare rule-based and evolved behavior
- separate training from visualization

---

## Core Architecture

The examples are separated into reusable components.

---

### Env

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

The environments support headless evaluation without active rendering.

---

### Controller

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

### Task

Tasks connect EvoLib individuals with environments.

Responsibilities:

- create environments
- create controllers
- evaluate individuals
- provide visualization helpers

Tasks are the main integration layer between EvoLib and the example
environments.

---

### Renderer

Renderers visualize environment state using Pygame.

Responsibilities:

- drawing
- overlays
- sensor visualization
- debug output

Rendering is separated from simulation so that training can run headless.

---

### Checkpoints

Training results are stored as checkpoints.

Typical checkpoint contents:

- evolved individual
- environment name
- optional difficulty or environment parameters
- random seed

This allows trained controllers to be loaded and visualized separately from
training.

---

## Optional Difficulty Presets

Some examples may provide difficulty presets such as:

- easy
- medium
- hard

---

## Debug Visualization

Training can optionally visualize the current best individual.

Typical use cases:

- debugging reward functions
- inspecting controller behavior
- checking sensor and collision behavior
- observing whether training produces useful behavior

Example:

```bash
python jumper_train.py --debug
```

---

## GIF Export

Some examples support GIF export during training.

This is useful for:

- documenting behavior
- comparing generations
- inspecting regressions after code changes

Example output:

```text
frames/gen_025.gif
```

---

## Current Examples

| Environment | Focus |
|---|---|
| LineFollower | steering and sensor feedback |
| Jumper | jump timing and action strength |
| Collector | co-evolution of perception and control |

---

## Example Philosophy

The examples intentionally avoid unnecessary complexity.

The focus is on:

- behavior
- observations
- rewards
- evolution

Small environments are easier to reason about, easier to debug, and easier to
adapt for experiments.

---

## Planned Examples

| Environment | Core concept |
|---|---|
| Collector | small 2D food-collection task |
| MemoryTask | recurrent memory and temporal behavior |
| PredatorPrey | co-evolution and interacting agents |

---

## Technical Direction

Possible future improvements include:

- BatchEnv integration
- parallel evaluation
- improved logging and tracking


---

## Requirements

Minimum requirements:

- Python 3.12+
- pygame

Install pygame:

```bash
pip install pygame
```
