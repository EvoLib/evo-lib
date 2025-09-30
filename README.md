# EvoLib – A Modular Framework for Evolutionary Computation

[![Docs Status](https://readthedocs.org/projects/evolib/badge/?version=latest)](https://evolib.readthedocs.io/en/latest/)
[![Code Quality & Tests](https://github.com/EvoLib/evo-lib/actions/workflows/ci.yml/badge.svg)](https://github.com/EvoLib/evo-lib/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI version](https://badge.fury.io/py/evolib.svg)](https://pypi.org/project/evolib/)
[![Project Status: Beta](https://img.shields.io/badge/status-beta-blue.svg)](https://github.com/EvoLib/evo-lib)

<p align="center">
  <img src="https://raw.githubusercontent.com/EvoLib/evolib/main/assets/evolib_256.png" alt="EvoLib Logo" width="256"/>
</p>

EvoLib is a lightweight and transparent framework for evolutionary computation, focusing on simplicity, modularity, and clarity — aimed at experimentation, teaching, and small-scale research rather than industrial-scale applications.

---

## Key Features

- **Transparent design**: configuration via YAML, type-checked validation, and clear module boundaries.  
- **Modularity**: mutation, selection, crossover, and parameter representations can be freely combined.  
- **Educational value**: examples and a clean API make it practical for illustrating evolutionary concepts.  
- **Neuroevolution support**: structural mutations (adding/removing neurons and connections) and evolvable networks via EvoNet.  
- **Parallel evaluation (optional)**: basic support for [Ray](https://www.ray.io/) to speed up fitness evaluations.  
- **Type-checked**: PEP8 compliant, and consistent code style.  


> **EvoLib is currently in beta. The core API and configuration format are stable, but some features are still under development.**

---
<p align="center">
  <img src="https://raw.githubusercontent.com/EvoLib/evo-lib/main/examples/05_advanced_topics/04_frames_vector_obstacles/04_vector_control_obstacles.gif" alt="Sample Plot" width="512"/>
</p>

---

## Installation

```bash
pip install evolib
```

Requirements: Python 3.10+ and packages in `requirements.txt`.

---


## Example Usage

```python
from evolib import Pop

def my_fitness(indiv):
    # Custom fitness function (example: sum of vector)
    indiv.fitness = sum(indiv.para["main"].vector)

pop = Pop(config_path="config/my_experiment.yaml",
          fitness_function=my_fitness)

# Run the evolutionary process
pop.run()
```

For full examples, see 📁[`examples/`](https://github.com/EvoLib/evo-lib/tree/main/examples) – including adaptive mutation, controller evolution, and network approximation.

---

# Configuration Example (YAML)

A core idea of EvoLib is that experiments are defined entirely through YAML configuration files.
This makes runs explicit, reproducible, and easy to adapt. The example below demonstrates
different modules (vector + EvoNet) with mutation, structural growth, and stopping criteria.


```yaml
parent_pool_size: 20
offspring_pool_size: 60
max_generations: 100
num_elites: 2
max_indiv_age: 0

stopping:
  target_fitness: 0.01
  patience: 20
  min_delta: 0.0001
  minimize: true

evolution:
  strategy: mu_comma_lambda

modules:
  controller:
    type: vector
    dim: 8
    initializer: normal_vector
    bounds: [-1.0, 1.0]
    mutation:
      strategy: adaptive_individual
      probability: 1.0
      strength: 0.1

  brain:
    type: evonet
    dim: [4, 6, 2]
    activation: [linear, tanh, tanh]
    initializer: normal_evonet
    mutation:
      strategy: constant
      probability: 1.0
      strength: 0.05

      # Optional fine-grained control
      activations:
        probability: 0.01
        allowed: [tanh, relu, sigmoid]

      structural:
        add_neuron: 0.01
        add_connection: 0.05
        remove_connection: 0.02
        recurrent: local  # none | direct | local | all
        keep_connected: true

```

---

> ℹ️ Multiple parameter types (e.g. vector + evonet) can be combined in a single individual. Each component evolves independently, using its own configuration.

---

## Documentation 

Documentation for EvoLib is available at: 👉 https://evolib.readthedocs.io/en/latest/

---


## Use Cases

EvoLib is developed for clarity, modularity, and exploration in evolutionary computation.  
It can be applied to:

- **Illustrating concepts**: simple, transparent examples for teaching and learning.  
- **Neuroevolution**: evolve weights and network structures using EvoNet.  
- **Multi-module evolution**: combine different parameter types (e.g. controller + brain).  
- **Strategy comparison**: benchmark and visualize mutation, selection, and crossover operators.  
- **Function optimization**: test behavior on benchmark functions (Sphere, Ackley, …).  
- **Showcases**: structural XOR, image approximation, and other demo tasks.  
- **Rapid prototyping**: experiment with new evolutionary ideas in a lightweight environment.  

---

## Preview: Pygame Integration

Early prototypes demonstrate how evolutionary algorithms can evolve both neural networks and sensor properties such as number, range, and orientation for agents in 2D worlds built with pygame. This illustrates how networks and sensors co-adapt to dynamic environments with collisions and feedback.

### Ant/Food Prototype
In this video, agents use simple sensors to learn how to collect food while avoiding collisions with the environment.

<p align="center">
  <img src="https://raw.githubusercontent.com/EvoLib/evo-lib/main/assets/ant.gif" alt="Pygame Integration Preview" width="640"/>
</p>

### Flappy Bird–style Prototype
Another prototype uses a **Flappy Bird–like 2D world**, where agents must pass through moving gaps.
Both the **neural controller** and the **sensors** (number, length, angle) are evolved, allowing perception and action to adapt together.
This illustrates how EvoLib can be applied to simple game-like environments, making the joint evolution of sensing and control directly observable.

<p align="center">
  <img src="https://raw.githubusercontent.com/EvoLib/evo-lib/main/assets/flappy.gif" alt="Pygame Integration Preview" width="160"/>
</p>

*This video shows the best agent from the final generation rather than the full evolutionary process.*

---

## Learn EvoLib in 5 Steps

EvoLib includes a small set of examples that illustrate the core concepts step by step:

1. [Hello Evolution](examples/01_basic_usage/04_fitness.py) – minimal run with a custom fitness function and visible improvement over generations.
2. [Strategies in Action](examples/02_strategies/03_mu_lambda.py) – (μ + λ) evolution step by step.
3. [Function Approximation](examples/04_function_approximation/02_sine_point_approximation.py) – evolve support points to match a sine curve.
4. [Evolution as Control](examples/05_advanced_topics/04_vector_control_with_obstacles.py) – evolve a controller in an environment.
5. [Neuroevolution with Structural Growth](examples/07_evonet/03_structural_xor.py) – evolve networks with growing topology.

For deeper exploration, see the [full examples directory](examples/)

---

## Roadmap

- [X] Adaptive Mutation (global, individual, per-parameter)
- [X] Flexible Crossover Strategies (BLX, intermediate, none)
- [X] Structured Neural Representations (EvoNet)
- [X] Composite Parameters (multi-module individuals)
- [X] Neuroevolution
- [X] Topological Evolution (neurons, edges)
- [X] Ray Support for Parallel Evaluation (early prototypes)
- [ ] Advanced Visualization
- [ ] Game Environment Integration (pygame, PettingZoo - early prototypes)


---


## License

MIT License – see [MIT License](https://github.com/EvoLib/evo-lib/tree/main/LICENSE).
