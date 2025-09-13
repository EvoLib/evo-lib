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
| `dim`         | list | —       | Layer sizes, e.g. `[4, 6, 2]`.                                |
| `activation`  | list | —       | Activation functions per layer (e.g. `[linear, tanh, tanh]`). |
| `initializer` | str  | —       | Network initialization method (e.g. `normal_evonet`).         |
| `mutation`    | dict | —       | Mutation settings (weights, biases, structure).               |

Example:

```yaml
modules:
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
        recurrent: local  # none | direct | local | all
        keep_connected: true
```

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

