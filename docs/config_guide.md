# Configuration Guide

EvoLib experiments are configured with YAML files. A configuration defines the
population, evolutionary strategy, and one or more evolvable parameter modules.

This guide focuses on common configurations. For the complete list of fields,
defaults, and constraints, see the
[Configuration Parameters](config_parameter.md) reference.

## Basic Structure

A small configuration usually contains:

- population sizes and the generation limit,
- an evolutionary strategy,
- one or more modules under `modules`.

The following configuration defines a two-dimensional vector optimized with a
(μ + λ) strategy:

```yaml
parent_pool_size: 20
offspring_pool_size: 40
max_generations: 100
num_elites: 2

evolution:
  strategy: mu_plus_lambda

modules:
  parameters:
    type: vector
    dim: 2
    initializer: uniform
    bounds: [-5.0, 5.0]
    mutation:
      strategy: constant
      probability: 1.0
      strength: 0.1
```

A fitness function can then be attached to the population:

```python
import numpy as np

from evolib import Individual, Population


def sphere_fitness(indiv: Individual) -> float:
    values = np.asarray(indiv.para["parameters"].vector, dtype=float)
    return float(np.sum(values**2))


pop = Population(
    config_path="population.yaml",
    fitness_function=sphere_fitness,
)
pop.run()
```

Fitness functions may return a numeric fitness value. Assigning
`indiv.fitness` directly is also supported.

## Selection, Replacement, and Stopping

Additional operators can be configured independently. For example, a flexible
strategy can use tournament selection and steady-state replacement:

```yaml
parent_pool_size: 40
offspring_pool_size: 80
max_generations: 120
num_elites: 4

stopping:
  target_fitness: 0.001
  patience: 20

evolution:
  strategy: flexible

selection:
  strategy: tournament
  tournament_size: 3

replacement:
  strategy: steady_state
  num_replace: 5

modules:
  parameters:
    type: vector
    dim: 6
    initializer: uniform
    bounds: [-1.0, 1.0]
    mutation:
      strategy: adaptive_individual
      probability: 0.8
      min_strength: 0.01
      max_strength: 0.05
```

The available strategies and their strategy-specific parameters are listed in
the parameter reference.

## EvoNet Configuration

An EvoNet module defines the network dimensions, initial connectivity, parameter
initialization, and evolutionary operators.

This example starts with a small feedforward network:

```yaml
parent_pool_size: 20
offspring_pool_size: 40
max_generations: 200
num_elites: 0

evolution:
  strategy: mu_plus_lambda

modules:
  brain:
    type: evonet
    dim: [2, 4, 1]
    activation: [linear, tanh, linear]
    initializer: default

    connectivity:
      scope: adjacent
      density: 1.0
      recurrent: none

    weights:
      initializer: normal
      std: 0.5
      bounds: [-5.0, 5.0]

    bias:
      initializer: zero
      bounds: [-1.0, 1.0]

    mutation:
      strategy: constant
      probability: 1.0
      strength: 0.05
```

`scope` determines which feedforward connections are allowed during
initialization, while `density` determines how many of those connections are
created. `recurrent: none` disables recurrent connections.

## Structural Evolution

Structural mutation can be added to the EvoNet `mutation` block. The following
fragment extends the previous network with neuron and connection mutation:

```yaml
mutation:
  strategy: constant
  probability: 1.0
  strength: 0.05

  structural:
    add_neuron:
      probability: 0.01
      init_connection_ratio: 0.5
      activations_allowed: [tanh]
      init: random

    remove_neuron:
      probability: 0.01

    add_connection:
      probability: 0.05
      max: 2
      init: random

    remove_connection:
      probability: 0.05
      max: 2

    topology:
      recurrent: none
      connection_scope: crosslayer
      max_neurons: 20
      max_connections: 50
```

The topology block constrains structural growth. It does not replace the
`connectivity` block used to build the initial network.

## Recurrent Connections and Delays

Recurrent connection kinds are enabled in `connectivity.recurrent`. Delays can
then be initialized and mutated separately:

```yaml
connectivity:
  scope: adjacent
  density: 1.0
  recurrent: [direct]

delay:
  initializer: uniform
  bounds: [1, 8]

mutation:
  strategy: constant
  probability: 1.0
  strength: 0.05

  delay:
    probability: 0.05
    mode: delta_step
    delta: 1
    bounds: [1, 16]
```

The top-level `delay` block initializes delays when the network is built.
`mutation.delay` controls how existing recurrent delays change during
evolution.

## Parallel Evaluation

Fitness evaluation can optionally use Ray. Install the parallel extra first:

```bash
pip install "evolib[parallel]"
```

Then select the Ray backend:

```yaml
parallel:
  backend: ray
  num_cpus: 4
```

If `num_cpus` is omitted, Ray selects the available resources. The optional
`address` field is passed directly to `ray.init()` when connecting to a Ray
cluster.

Without a `parallel` block, fitness evaluation is sequential.

## Further Examples

Runnable examples are available in the
[EvoLib examples](https://github.com/EvoLib/evo-lib/tree/main/examples).

For all available fields and constraints, see
[Configuration Parameters](config_parameter.md).
