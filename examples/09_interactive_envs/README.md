# Interactive Environments (EvoLib)

This module provides **interactive, visual environments** for EvoLib.

The goal is to make evolutionary algorithms and neural networks
**intuitive and observable**.

---

## Concept

These environments focus on:

- visual feedback
- step-by-step understanding
- interactive exploration

Each environment follows the same pattern:

    play --> rule --> train --> watch

---

## Learning Pipeline

### 1. Play (Manual Control)

- user controls the agent
- builds intuition

### 2. Rule-Based Solution

- simple heuristic controller
- shows that the problem is solvable

### 3. Train (Evolution)

- EvoLib evolves a neural network
- behavior improves over generations

### 4. Watch

- visualize the best individual
- observe emergent behavior

---

## Architecture

Core interfaces:

class Env:
    def reset(self, seed=None): ...
    def step(self, action): ...

class Controller:
    def act(self, observation): ...

Evaluation:

reward = evaluate_episode(env, controller)

---


## Environments

| Example | Description |
|--------|------------|
| 01_line_follower | follow a line using two sensors |

---

## Requirements

- Python 3.10+
- pygame

Install:

    pip install pygame

