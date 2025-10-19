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

| Parameter | Type | Default | Description |
|------------|------|----------|-------------|
| `max_new_connections` | int | 1 | Maximum number of new connections that can be created in a single structural mutation. |
| `max_removed_connections` | int | 1 | Maximum number of existing connections that can be removed per mutation. |
| `connection_scope` | str | `adjacent` | Defines which layers may connect: `adjacent` (only neighboring layers) or `crosslayer` (any-to-any). |
| `connection_init` | str | `near_zero` | Defines how weights of new connections are initialized: `random`, `zero`, `near_zero`, or `none`. |
| `connection_init_value` | float \| null | null | If set, overrides `connection_init` with a fixed numeric weight value. |
| `keep_connected` | bool | true | Ensures neurons remain connected after structural mutation (avoids isolated nodes). |
| `recurrent` | str | `none` | Enables optional recurrence: `none`, `direct`, `local`, or `all`. |
| `max_nodes` | int | 0 | Upper limit for the total number of neurons in the network (`0` = unlimited). |
| `max_edges` | int | 0 | Upper limit for the total number of connections (`0` = unlimited). |

---

**Notes**

- Structural mutations are triggered according to their probabilities in the `mutation.structural` block.
- `connection_scope` and `connection_init` provide fine control over how new edges integrate into the topology.
- Using `near_zero` for new connections can reduce disruption and stabilize early learning phases.
- Growth limits (`max_nodes`, `max_edges`) are optional safety guards for open-ended topologies.

---

## Mutation Settings

Mutation can be specified per module. Parameters vary by strategy, but common fields are:

| Parameter     | Type  | Default | Explanation                                                 |
| ------------- | ----- | ------- | ----------------------------------------------------------- |
| `strategy`    | str   | —       | Mutation strategy (e.g. `constant`, `adaptive_individual`). |
| `probability` | float | 1.0     | Probability of mutating a given parameter.                  |
| `strength`    | float | —       | Standard deviation / scale of the mutation.                 |

---

## Initializers

Initializers define how parameters are set at creation (before any evolution).
Available initializers depend on the module type:

* **Vectors**: `normal_vector`, `random_vector`, `zero_vector`, `fixed_vector`, `adaptive_vector`
* **EvoNet**: `normal_evonet`

---

## Putting It All Together

A complete configuration may combine several modules:

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
        add_neuron: 0.01
        add_connection: 0.05
        remove_connection: 0.02
        recurrent: local
        keep_connected: true
```

