# Configuration Parameters

This guide provides an overview of the configuration parameters available in EvoLib.
Configurations are written in **YAML** and passed to a `Pop` / `Population` instance.

The YAML defines:
- run control (pools, stopping, strategy),
- operators (selection, replacement, mutation, crossover),
- modules (vector / evonet search spaces).

---

## Global Parameters

| Parameter            | Type         | Default | Explanation                                                          |
|----------------------|--------------|---------|----------------------------------------------------------------------|
| `random_seed`        | int \| null  | null    | Global RNG seed for reproducibility.                                 |
| `parent_pool_size`   | int          | —       | Number of parents selected for the next generation (μ).              |
| `offspring_pool_size`| int          | —       | Number of offspring generated each generation (λ).                   |
| `max_generations`    | int          | —       | Maximum number of generations before termination.                    |
| `num_elites`         | int          | 0       | Number of top individuals copied unchanged into the next generation. |
| `max_indiv_age`      | int          | 0       | Maximum age of individuals (0 = no age limit).                       |

---

## Logging

| Parameter         | Type | Default | Explanation                                  |
|------------------|------|---------|----------------------------------------------|
| `lineage`| bool | false   | Enable lineage logging (if supported by run).|

Example:

```yaml
logging:
  lineage: true
```

---

## Parallelization Settings

Optional parameters to enable parallel evaluation of individuals.

| Parameter    | Type | Default | Explanation                                                                 |
|--------------|------|---------|-----------------------------------------------------------------------------|
| `backend`    | str  | none    | Parallel backend (`ray` or `none`).                                         |
| `num_cpus`   | int  | 1       | Number of logical CPUs Ray may use for evaluation (local mode).             |
| `address`    | str  | auto    | `"auto"` = local Ray; or `ray://host:port` for connecting to a remote Ray cluster. |

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

| Parameter        | Type          | Default | Explanation                                                     |
|------------------|---------------|---------|-----------------------------------------------------------------|
| `target_fitness` | float \| null | null    | Stop once best fitness reaches this threshold.                  |
| `patience`       | int \| null   | null    | Allow this many generations without improvement before stop.    |
| `min_delta`      | float         | 0.0     | Minimum improvement considered as progress.                     |
| `minimize`       | bool          | true    | Whether the target fitness is minimized (default) or maximized. |
| `time_limit_s`   | float \| null | null    | Wallclock time limit in seconds (hard stop).                    |

Example:

```yaml
stopping:
  target_fitness: 0.01
  patience: 20
  min_delta: 0.0001
  minimize: true
  time_limit_s: 30.0
```

---

## Evolution Settings

| Parameter   | Type | Default | Explanation                                                                  |
|-------------|------|---------|------------------------------------------------------------------------------|
| `strategy`  | str  | —       | The evolutionary strategy to use (e.g. `mu_comma_lambda`, `mu_plus_lambda`). |

Example:

```yaml
evolution:
  strategy: mu_comma_lambda
```

---

# HELI — Hierarchical Evolution with Lineage Incubation

HELI creates temporary subpopulations for structurally mutated individuals and evolves them locally before reintegration.

## Parameters

| Parameter             | Type          | Default | Explanation                                                              |
|-----------------------|---------------|---------|--------------------------------------------------------------------------|
| `generations`         | int           | —       | Number of local generations performed inside HELI.                       |
| `offspring_per_seed`  | int           | —       | Number of offspring per structural mutant (seed).                        |
| `max_fraction`        | float         | 1.0     | Maximum ratio of active HELI subpopulations.                             |
| `reduce_sigma_factor` | float         | 1.0     | Scaling factor applied to mutation strength during HELI evolution.       |
| `drift_stop_above`    | float \| null | null    | Abort incubation if drift exceeds this value (seed too poor).            |
| `drift_stop_below`    | float \| null | null    | Abort incubation if drift goes below this value (seed already good).     |

## Notes

- HELI runs **only** when a structural mutation occurs.
- HELI inherits the module mutation configuration, scaled by `reduce_sigma_factor`.

Example:

```yaml
evolution:
  strategy: mu_plus_lambda
  heli:
    generations: 10
    offspring_per_seed: 8
    max_fraction: 1.0
    reduce_sigma_factor: 0.5
    drift_stop_above: 2.0
    drift_stop_below: -0.25
```

---

## Modules

Modules define the parameter representation(s) of each individual. Multiple modules can be combined.
Depending on `structure`, the vector can represent a simple flat genome or a
structured parameter layout (e.g. network-like interpretation).

---

### Vector Module

The `vector` module defines an evolvable parameter vector.

| Parameter                        | Type                                              | Default        | Description |
|----------------------------------|---------------------------------------------------|----------------|------------|
| `type`                           | `"vector"`                                        | —              | Module type identifier. |
| `dim`                            | `int \| list[int]`                                | —              | Vector length (`int`) or structured dimensions (`list[int]`). Must be > 0. |
| `structure`                      | `"flat" \| "net"                                  | `"flat"`       | Structural interpretation of the vector. |
| `initializer`                    | `str`                                             | —              | Initializer name from the registry (e.g. `uniform`, `zero`, `normal`, `fixed`). |
| `bounds`                         | `tuple[float, float]`                             | `[-1.0, 1.0]`  | Hard clamp range applied after mutation. |
| `init_bounds`                    | `tuple[float, float] \| null`                     | `null`         | Clamp applied only during initialization. Falls back to `bounds` if not set. |
| `shape`                          | `tuple[int, ...] \| null`                         | `null`         | Optional explicit shape. If set, `dim = product(shape)`. Shape is retained as metadata. |
| `values`                         | `list[float] \| null`                             | `null`         | Required for `initializer: fixed_vector`. If `dim` is omitted, it is inferred from `len(values)`. |
| `activation`                     | `str \| null`                                     | `null`         | Only relevant if `structure: "net"`. |
| `mean`                           | `float \| null`                                   | `0.0`          | Mean parameter for normal-based initializers (if used by initializer). |
| `std`                            | `float \| null`                                   | `1.0`          | Standard deviation for normal-based initializers (if used by initializer). |
| `mutation`                       | `dict`                                            | —              | **Required.** Mutation configuration (see below). |
| `randomize_mutation_strengths`   | `bool \| null`                                    | `false`        | If true, per-parameter strengths are randomly initialized within min/max bounds (strategy-dependent). |
| `tau`                            | `float \| null`                                   | `0.0`          | Scaling factor for self-adaptive mutation strategies. Interpretation is strategy-dependent. |
| `crossover`                      | `dict \| null`                                    | `null`         | Optional crossover configuration. Semantics depend on operator. |


## Mutation Configuration

The `mutation` block must follow the `MutationConfig` schema.

### Common Fields

| Parameter       | Type     | Description |
|-----------------|----------|-------------|
| `strategy`      | `str`    | Mutation strategy (e.g. `constant`, `adaptive_individual`, `adaptive_global`). |
| `probability`   | `float`  | Mutation probability.|
| `strength`      | `float`  | Mutation strength.|

---

### Strategy: `constant`

Uses a fixed mutation strength.

Requires:
- `probability`
- `strength`

```yaml
mutation:
  strategy: constant
  probability: 1.0
  strength: 0.05
```

### Strategy: `adaptive_individual`

Self-adaptive mutation at the individual level.

Requires:
- `probability`
- `min_strength`
- `max_strength`

```yaml
mutation:
  strategy: adaptive_individual
  probability: 1.0
  min_strength: 0.01
  max_strength: 0.05
```

### Strategy: `adaptive_global`

Global self-adaptive mutation strength shared across parameters.

```yaml
mutation:
  strategy: adaptive_global
  probability: 1.0
  min_strength: 0.01
  max_strength: 0.1
```


### Minimal Example:

```yaml
modules:
  main:
    type: vector
    dim: 8
    initializer: uniform
    bounds: [-1.0, 1.0]

    mutation:
      strategy: constant
      probability: 1.0
      strength: 0.05
```

### Fixed Vector Example
```yaml
modules:
  main:
    type: vector
    initializer: fixed_vector
    values: [0.0, 1.0, 0.5, -0.5]
    bounds: [-1.0, 1.0]

    mutation:
      strategy: constant
      probability: 1.0
      strength: 0.01
```
If dim is omitted, it is inferred from values.

### Advanced Example (Self-Adaptive)
```yaml
modules:
  main:
    type: vector
    structure: flat
    dim: 6
    initializer: normal_vector
    mean: 0.0
    std: 0.2
    bounds: [-1.0, 1.0]

    mutation:
      strategy: adaptive_individual
      probability: 0.8
      min_strength: 0.01
      max_strength: 0.05

    tau: 0.0
    randomize_mutation_strengths: false
```
---

### EvoNet Module

#### Core fields

| Parameter          | Type                | Default | Explanation |
|-------------------|---------------------|---------|-------------|
| `dim`             | list[int]           | —       | Layer sizes, e.g. `[4, 0, 0, 2]`. Hidden layers can start empty (0) and grow through structural mutation. |
| `activation`      | str \| list[str]    | —       | If list: activation per layer. If str: used for non-input layers; input layer is treated as linear. |
| `initializer`     | str                 | default | Topology preset (e.g. `default`, `unconnected`, `identity`). Parameter initialization is configured via `weights`, `bias`, and `delay`. |
| `connectivity`    | `dict`                   | —       | **Required.** Defines feedforward scope/density and allowed recurrent kinds. |
| `weights`         | dict                | —       | Weight init and bounds configuration (initializer, bounds, optional params). | 
| `bias`            | dict                | —       | Bias init and bounds configuration (initializer, bounds, optional params). | 
| `neuron_dynamics` | list[dict] \| null  | null    | Optional per-layer neuron dynamics specification. Must match `len(dim)`. |
| `mutation`        | dict \| null        | null    | Mutation settings for weights, biases, activations, delay, and structure. |
| `crossover`       | dict \| null        | null    | Optional crossover settings (weight/bias level). |

#### Connectivity (`connectivity`)

| Field       | Type                         | Default | Notes |
|------------|------------------------------|---------|------|
| `scope`    | `"adjacent" \| "crosslayer"` | —       | **Required.** Allowed feedforward edge scope at initialization. |
| `density`  | `float`                      | —       | **Required.** Fraction of allowed feedforward edges created at init `(0,1]`. |
| `recurrent`| `list[str]`                  | `[]`    | Allowed recurrent edge kinds. Empty list means none. |

Example:

```yaml
connectivity:
  scope: crosslayer
  density: 0.5
  recurrent: [direct]
```

##### weights block
|Parameter	 | Type	 | Default	 | Explanation |
|------------|-------|-----------|-------------|
|initializer | str   | "normal"	 | Weight initializer preset (normal, uniform, zero, …). |
|bounds	     | list[float]	| [-0.5, 0.5]	| Hard clipping bounds. |
|std         | float | null	|null | Std-dev for normal (if used). |

##### bias block
|Parameter	 | Type	 | Default	 | Explanation |
|------------|-------|-----------|-------------|
|initializer | str   | "normal"	 | Bias initializer preset (normal, uniform, zero, …). |
|bounds	     | list[float]	| [-1.0, 1.0]	| Hard clipping bounds. |
|std         | float | null	| Std-dev for normal (if used). |

---

#### EvoNet Initializer (Topology Presets)

`initializer` selects a **topology preset**.  

Parameter initialization is configured via:
- `weights`
- `bias`
- `delay` (recurrent connections only)

Allowed presets:

| Initializer            | Meaning (topology only) |
|------------------------|-------------------------|
| `default`       | Standard EvoNet topology preset (uses `connection_scope`, `connection_density`, and `recurrent`). |
| `unconnected`   | Creates neurons/layers but starts with **no connections** (use structural mutation to grow). |
| `identity`      | Special preset intended for stable recurrent memory (may override parameters internally; see notes below). |

---


#### Activation: special modes

You can also use `activation: random` with an allowed set (if supported by your config schema):

```yaml
modules:
  brain:
    type: evonet
    dim: [2, 0, 0, 1]
    activation: random
    activations_allowed: [tanh, relu, sigmoid]
```

#### Neuron dynamics example

```yaml
modules:
  brain:
    type: evonet
    dim: [1, 16, 1]
    activation: [linear, tanh, sigmoid]
    neuron_dynamics:
      - name: standard
        params: {}
      - name: leaky
        params: {alpha: 0.9}
      - name: standard
        params: {}
```

---

### EvoNet Delay (recurrent edges)

EvoNet supports explicit integer delays on **recurrent connections**.

There are two distinct configuration knobs:

1) `delay:` initializes delays at build time (recurrent edges only).
2) `mutation.delay:` mutates delays during evolution.

#### Delay initialization

```yaml
modules:
  brain:
    type: evonet
    ...
    delay:
      initializer: uniform  # uniform | fixed
      bounds: [1, 8]
      # value: 3            # only for fixed
```

---

## EvoNet Mutation Parameters

Mutation is specified inside `modules.<name>.mutation`.

### Common EvoNet mutation fields (weights)

| Parameter      | Type  | Default | Explanation |
|---------------|-------|---------|-------------|
| `strategy`    | str   | —       | Mutation strategy (e.g. `constant`, `adaptive_individual`, ...). |
| `probability` | float | —       | Probability to mutate (per individual per generation). |
| `strength`    | float | —       | Mutation strength (sigma / step size, strategy-dependent). |

### Bias override (optional)

`mutation.biases` overrides the global EvoNet mutation parameters for biases only.

```yaml
mutation:
  strategy: constant
  probability: 1.0
  strength: 0.05

  biases:
    strategy: constant
    probability: 0.8
    strength: 0.03
```

### Activation mutation (optional)

```yaml
mutation:
  ...
  activations:
    probability: 0.01
    allowed: [tanh, relu, sigmoid, elu, linear, linear_max1]
```

### Delay mutation (optional, recurrent edges only)

```yaml
mutation:
  ...
  delay:
    probability: 0.05
    bounds: [1, 16]
    mode: delta_step     # delta_step | resample
    delta: 1
```

---

## EvoNet — Structural Mutation Parameters

Structural mutations are part of the `mutation.structural` block inside an EvoNet module.
Four operator groups are available:

- **Add Neuron**
- **Remove Neuron**
- **Add Connection**
- **Remove Connection**

Topology-level constraints are defined inside `mutation.structural.topology`.

### Structural Mutation Operators

#### Add Neuron

```yaml
structural:
  add_neuron:
    probability: 0.015
    init_connection_ratio: 0.5
    activations_allowed: [tanh]
    init: random
```

#### Remove Neuron

```yaml
structural:
  remove_neuron:
    probability: 0.015
```

#### Add Connection

```yaml
structural:
  add_connection:
    probability: 0.05
    max: 3
    init: random
```

#### Remove Connection

```yaml
structural:
  remove_connection:
    probability: 0.05
    max: 3
```

### Topology constraints

| Field              | Type                           | Default     | Description |
|-------------------|--------------------------------|-------------|-------------|
| `recurrent`        | `list[str]`                    | `[]`        | Allowed recurrent kinds for `add_connection`. Empty list means none. |
| `connection_scope` | `"adjacent" \| "crosslayer" \| null` | `"adjacent"` | Constraint for structural add-neuron / add-connection edge placement. |
| `max_neurons`      | `int \| null`                  | `null`      | Upper bound on total neurons allowed (implementation-defined counting). |
| `max_connections`  | `int \| null`                  | `null`      | Upper bound on total connections allowed. |

---

## Full Example

```yaml
random_seed: 42

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
  heli:
    generations: 10
    offspring_per_seed: 8
    max_fraction: 1.0
    reduce_sigma_factor: 0.5

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
      min_strength: 0.01
      max_strength: 0.05

  brain:
    type: evonet
    dim: [4, 0, 0, 2]
    activation: [linear, tanh, tanh, tanh]

    delay:
      initializer: uniform
      bounds: [1, 8]

    mutation:
      strategy: constant
      probability: 1.0
      strength: 0.05

      activations:
        probability: 0.01
        allowed: [tanh, relu, sigmoid]

      delay:
        probability: 0.05
        bounds: [1, 16]
        mode: delta_step
        delta: 1

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
