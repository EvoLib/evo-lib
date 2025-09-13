# Configuration Guide

EvoLib experiments are defined via **YAML configuration files**.  
This makes setups explicit, reproducible, and easy to adapt.

Below you find three representative configurations:

- **A minimal example** – the smallest viable run.  
- **A full vector-based setup** – showing multiple modules, selection, and replacement.  
- **A full EvoNet setup** – demonstrating weight and structural mutation.  

Each snippet is commented inline for clarity.

---

## A) Minimal configuration

This is the smallest possible (μ + λ) run.  
It demonstrates the core building blocks: pools, a single vector module, and constant mutation.  
A fitness function (e.g. `sphere`) must still be provided in practice.

```yaml
# Minimum configuration — smallest viable building block for a (μ + λ) run.
parent_pool_size: 2         # μ: number of parents kept
offspring_pool_size: 4      # λ: number of offspring produced
max_generations: 10         # hard stop
num_elites: 0               # no elitism

evolution:
  strategy: mu_plus_lambda  # classic (μ + λ) evolution strategy

modules:
  test-vector:              # logical module name
    type: vector
    initializer: random_vector
    dim: 2
    bounds: [-1.0, 1.0]

    mutation:
      strategy: constant    # fixed-strength perturbation
      strength: 0.01
      probability: 1.0
```

---

## B) Maximum Vector configuration

This example shows a more complete setup: tournament selection, steady-state replacement, stopping criteria, and two vector modules with different operator settings.

```yaml

random_seed: 42             # reproducibility

parent_pool_size: 40
offspring_pool_size: 80
max_generations: 120
max_indiv_age: 2
num_elites: 4

stopping:
  target_fitness: 0.001

evolution:
  strategy: flexible

selection:
  strategy: tournament
  tournament_size: 3

replacement:
  strategy: steady_state
  num_replace: 5

modules:
  xs:
    type: vector
    dim: 6
    initializer: random_vector
    bounds: [0.0, 6.283185307]   # [0, 2π]
    mutation:
      strategy: adaptive_individual
      probability: 0.8
      min_strength: 0.01
      max_strength: 0.05

  ys:
    type: vector
    dim: 6
    initializer: zero_vector
    bounds: [-1.5, 1.5]
    mutation:
      strategy: constant
      probability: 0.8
      strength: 0.06
    crossover:
      strategy: constant
      probability: 0.3
      operator: blx

```

---

## C) Maximum EvoNet configuration

This configuration demonstrates EvoNet evolution: weight mutation, bias-specific overrides, and structural mutation.
The `dim: [2, 0, 0, 1]` starts with minimal topology, letting structural mutation grow hidden nodes and edges.
`max_nodes` and `max_edges` keep growth bounded.

```yaml
parent_pool_size: 20
offspring_pool_size: 40
max_generations: 200
max_indiv_age: 0
num_elites: 0

stopping:
  target_fitness: 0.001

evolution:
  strategy: mu_plus_lambda

modules:
  brain:
    type: evonet
    dim: [2, 0, 0, 1]       # minimal topology; hidden layers start empty
    activation: [linear, tanh, tanh, sigmoid]
    initializer: normal_evonet
    weight_bounds: [-5.0, 5.0]
    bias_bounds:   [-1.0, 1.0]

    mutation:
      strategy: constant
      strength: 0.1
      probability: 0.9

      biases:
        strategy: constant
        strength: 0.05
        probability: 0.8

      structural:
        add_neuron: 0.03
        recurrent: "none"    # none | direct | local | all
        remove_neuron: 0.01
        add_connection: 0.05
        remove_connection: 0.01
        split_connection: 0.02

        max_nodes: 3
        max_edges: 20

```

---

## Further examples

For complete, runnable examples including fitness definitions and visualization,
please refer to the GitHub repository:

👉 [EvoLib Examples on GitHub](https://github.com/EvoLib/evo-lib/tree/main/examples)

For a complete list of all available configuration fields,
see the [Configuration Parameters](config_parameter.md) reference.
