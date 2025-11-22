# Configuration Parameters

This guide provides an overview of the configuration parameters available in EvoLib.
Configurations are written in **YAML** and passed to a `Population` instance.

The parameters are grouped into **global parameters**, **evolution strategy settings**, and **module-specific parameters**.

---

## Global Parameters

| Parameter             | Type | Default | Explanation                                                          |
| --------------------- | ---- | ------- | -------------------------------------------------------------------- |
| `parent_pool_size`    | int  | —       | Number of parents selected for the next generation.                  |
| `offspring_pool_size` | int  | —       | Number of offspring generated each generation.                       |
| `max_generations`     | int  | —       | Maximum number of generations before termination.                    |
| `num_elites`          | int  | 0       | Number of top individuals copied unchanged into the next generation. |
| `max_indiv_age`       | int  | 0       | Maximum age of individuals (0 = no age limit).                       |


## Parallelization Settings

Optional parameters to enable parallel evaluation of individuals.

| Parameter   | Type | Default | Explanation                                                                 |
| ----------- | ---- | ------- | --------------------------------------------------------------------------- |
| `backend`   | str  | none    | Parallel backend (`ray` or `none`).                                          |
| `num_cpus`  | int  | 1       | Number of logical CPUs Ray may use for evaluation.                          |
| `address`   | str  | auto    | `"auto"` = local Ray; or `ray://host:port` for connecting to a remote Ray cluster. |

Example:

```yaml
parallel:
  backend: ray
  num_cpus: 4
  address: auto
```

---

## Stopping Criteria

Stopping criteria can be defined to terminate runs early.

| Parameter        | Type  | Default | Explanation                                                     |
| ---------------- | ----- | ------- | --------------------------------------------------------------- |
| `target_fitness` | float | —       | Stop once best fitness reaches this threshold.                  |
| `patience`       | int   | —       | Allow this many generations without improvement before stop.    |
| `min_delta`      | float | 0.0     | Minimum improvement considered as progress.                     |
| `minimize`       | bool  | true    | Whether the target fitness is minimized (default) or maximized. |

Example:

```yaml
stopping:
  target_fitness: 0.01
  patience: 20
  min_delta: 0.0001
  minimize: true
```

---

## Evolution Settings

| Parameter  | Type | Default | Explanation                                                                  |
| ---------- | ---- | ------- | ---------------------------------------------------------------------------- |
| `strategy` | str  | —       | The evolutionary strategy to use (e.g. `mu_comma_lambda`, `mu_plus_lambda`). |

Example:

```yaml
evolution:
  strategy: mu_comma_lambda
```

---

## Modules

Modules define the parameter representation(s) of each individual. Multiple modules can be combined.

### Common Fields

| Parameter     | Type | Default | Explanation                                                 |
| ------------- | ---- | ------- | ----------------------------------------------------------- |
| `type`        | str  | —       | Type of parameter representation (`vector`, `evonet`, ...). |
| `initializer` | str  | —       | Initialization method for the module.                       |
| `bounds`      | list | —       | Lower and upper limits for values (only for vectors).       |

---

### Vector Module

| Parameter     | Type | Default | Explanation                                   |
| ------------- | ---- | ------- | --------------------------------------------- |
| `dim`         | int  | —       | Dimensionality of the vector.                 |
| `initializer` | str  | —       | Initialization method (e.g. `normal_vector`). |
| `mutation`    | dict | —       | Mutation settings for the vector.             |

Example:

```yaml
modules:
  main:
    type: vector
    dim: 8
    initializer: normal_vector
    bounds: [-1.0, 1.0]
    mutation:
      strategy: adaptive_individual
      probability: 1.0
      strength: 0.1
```

---

### EvoNet Module

| Parameter     | Type | Default | Explanation                                                   |
| ------------- | ---- | ------- | ------------------------------------------------------------- |
| `dim`         | list | —       | Layer sizes, e.g. [4, 0, 0, 2]. Hidden layers can start empty (0) and grow through structural mutation. |
| `activation`  | list | —       | Activation functions per layer (e.g. `[linear, tanh, tanh, linear]`). |
| `initializer` | str  | —       | Network initialization method (e.g. `normal_evonet`, unconnected_evonet).         |
| `mutation`    | dict | —       | Mutation settings for weights, biases, activations, and structure. |
| `weight_bounds | list | —       | [min_w, max_w] — hard clipping bounds for connection weights. |
| `bias_bounds | list | —       | [min_b, max_b] — hard clipping bounds for neuron biases. |

Example:

```yaml
modules:
  brain:
    type: evonet
    dim: [2, 0, 0, 1]            # starts with minimal topology
    activation: [linear, tanh, tanh, sigmoid]
    initializer: normal_evonet
    weight_bounds: [-5.0, 5.0]
    bias_bounds:   [-1.0, 1.0]

    mutation:
      strategy: constant
      probability: 1.0
      strength: 0.05

      biases:
        strategy: constant
        probability: 0.8
        strength: 0.03

      activations:
        probability: 0.01
        allowed: [tanh, relu, sigmoid, elu, linear, linear_max1]

      structural:
        # Structural mutation probabilities
        add_neuron: 0.01
        remove_neuron: 0.01
        add_connection: 0.05
        remove_connection: 0.02
        split_connection: 0.00

        # NEW: control how many edges are modified per mutation
        max_new_connections: 1
        max_removed_connections: 1

        # NEW: restrict or extend feedforward connectivity
        connection_scope: adjacent      # adjacent | crosslayer

        # NEW: how new connections are initialized
        connection_init: near_zero      # random | zero | near_zero | none
        connection_init_value: null     # optional explicit weight value override

        # Recurrent settings
        recurrent: none                 # none | direct | local | all
        keep_connected: true            # prevents isolated neurons

        # Optional topological growth limits
        max_nodes: 0                    # 0 = unlimited
        max_edges: 0                    # 0 = unlimited


```

### EvoNet - Structural Mutation Parameters

This table summarizes the structural mutation parameters available for **EvoNet modules** in EvoLib.
They control how neurons and connections are added, removed, or initialized during evolution.

## 1. Overview

Structural mutations are part of the `mutation.structural` block inside a module of type `evonet`.
Four operator groups are available:

- **Add Neuron**
- **Remove Neuron**
- **Add Connection**
- **Remove Connection**

Topology-level constraints (e.g., max number of neurons) are defined inside a separate `topology:` section.

---

# 2. Structural Mutation Operators

## 2.1 Add Neuron

| Field | Type | Default | Description |
|-------|-------|----------|-------------|
| `probability` | float | — | Probability of inserting a new neuron. |
| `activations_allowed` | list[str] | `[tanh]` | Set of allowed activation functions for the new neuron. |
| `init` | str | `random` | Defines how the neuron and its initial connections are initialized. |
| `init_connection_ratio` | float | `0.3` | Proportion of possible edges that should be created when the neuron is inserted. |

---

## 2.2 Remove Neuron

| Field | Type | Default | Description |
|-------|-------|----------|-------------|
| `probability` | float | — | Probability of removing an existing non-input neuron. |

---

## 2.3 Add Connection

| Field | Type | Default | Description |
|-------|-------|----------|-------------|
| `probability` | float | — | Probability of inserting a new connection. |
| `max` | int | `1` | Maximum number of new connections created during one mutation event. |
| `init` | str | `random` | Initialization method for the new connection's weight. |

---

## 2.4 Remove Connection

| Field | Type | Default | Description |
|-------|-------|----------|-------------|
| `probability` | float | — | Probability of removing an existing connection. |
| `max` | int | `1` | Maximum number of connections removed during one mutation event. |

---

# 3. Topology Constraints

Topology-related parameters define global rules for allowed edges and network size.

These values are defined inside:

```yaml
structural:
  topology:
    ...
```

| Field | Type | Default | Description |
|-------|-------|----------|-------------|
| `recurrent` | str | `none` | Controls recurrence: `none`, `direct`, `local`, or `all`. |
| `connection_scope` | str | `adjacent` | Allowed layer connectivity: `adjacent` (neighbor layers only) or `crosslayer` (any-to-any). |
| `max_neurons` | int \| null | `null` | Maximum number of non-input neurons (`null` = unlimited). |
| `max_connections` | int \| null | `null` | Maximum number of edges (`null` = unlimited). |

---

# 4. EvoNet Initializer

The EvoNet module uses:

| Initializer         | Weights                            | Biases                           | Notes                                       |
|---------------------|------------------------------------|----------------------------------|---------------------------------------------|
| `normal_evonet`     | Normal(0, 0.5)                     | Normal(0, 0.5)                   | Default initializer for general use         |
| `unconnected_evonet`| None                               | 0                                | For pure structural growth; empty topology  |
| `random_evonet`     | Random                             | Uniform(bias_bounds)             | For broader stochastic exploration           |
| `zero_evonet`       | 0                                  | 0                                | Deterministic baseline; debugging            |
| `identity_evonet`   | Small random                       | Small random                     | Designed for stable recurrent memory         |


---

# 5. Full Example (Current Valid Syntax)

```yaml
parent_pool_size: 20
offspring_pool_size: 60
max_generations: 100
num_elites: 2

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

      activations:
        probability: 0.01
        allowed: [tanh, relu, sigmoid]

      structural:

        add_neuron:
          probability: 0.015
          activations_allowed: [tanh]
          init: random
          init_connection_ratio: 0.3

        remove_neuron:
          probability: 0.015

        add_connection:
          probability: 0.05
          max: 3
          init: random

        remove_connection:
          probability: 0.05
          max: 3

        topology:
          recurrent: none
          connection_scope: crosslayer
          max_neurons: 25
          max_connections: 50
```

---

# 6. Notes

- Only **one** structural operator is executed per mutation event.
- `max_neurons` and `max_connections` provide soft caps to prevent uncontrolled growth.
- `connection_scope: crosslayer` allows long-range edges, enabling richer architectures but also increasing search space.
