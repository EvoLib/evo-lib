# Configuration Parameters

This page lists the configuration parameters accepted by EvoLib.
Configurations are written in YAML and passed to `Population`.

For usage-oriented examples, see the configuration guide.

## Global Parameters

| Parameter | Type | Default | Description |
| --- | --- | ---: | --- |
| `parent_pool_size` | int | --- | Number of parents retained in each generation. Must be greater than 0. |
| `offspring_pool_size` | int | --- | Number of offspring produced per generation. Must be greater than 0. |
| `max_generations` | int | --- | Maximum number of generations. Must be greater than 0. |
| `num_elites` | int | --- | Number of elite individuals preserved each generation. Must be between 0 and `parent_pool_size`. |
| `max_indiv_age` | int | `0` | Maximum individual age in generations. `0` disables aging. |
| `random_seed` | int \\| null | `null` | Global random seed. Use an integer for reproducible runs. |

## Evolution

The optional `evolution` block selects the high-level evolutionary
strategy.

| Parameter | Type | Default | Description |
| --- | --- | ---: | --- |
| `strategy` | str | --- | `mu_plus_lambda`, `mu_comma_lambda`, `steady_state`, or `flexible`. |
| `heli` | dict \\| null | `null` | Optional HELI configuration. |

### HELI

HELI incubates individuals produced by structural mutation before they
re-enter the main evolutionary process.

| Parameter | Type | Default | Description |
| --- | --- | ---: | --- |
| `generations` | int | `5` | Number of local generations per incubated seed. |
| `offspring_per_seed` | int | `10` | Number of offspring generated per incubated seed. |
| `max_fraction` | float | `0.1` | Maximum fraction of offspring eligible for incubation. Must be in `[0, 1]`. |
| `reduce_sigma_factor` | float | `0.5` | Factor applied to mutation strength during incubation. Must be non-negative. |
| `drift_stop_above` | float \\| null | `null` | Stop incubation if drift exceeds this threshold. |
| `drift_stop_below` | float \\| null | `null` | Stop incubation if drift falls below this threshold. |
| `seed_selection` | str | `fitness` | Selection of eligible seeds: `fitness`, `random`, or `none`. |

## Selection

The optional `selection` block configures parent selection.

| Parameter | Type | Default | Description |
| --- | --- | ---: | --- |
| `strategy` | str | --- | `tournament`, `roulette`, `rank_linear`, `rank_exponential`, `sus`, `boltzmann`, `truncation`, or `random`. |
| `num_parents` | int \\| null | `null` | Optional number of parents selected by the strategy. |
| `tournament_size` | int \\| null | `null` | Tournament size for tournament selection. |
| `exp_base` | float \\| null | `null` | Base used by exponential ranking selection. |
| `fitness_maximization` | bool | `false` | If true, higher fitness values are considered better. |

Only parameters used by the selected strategy need to be specified.

## Replacement

The optional `replacement` block configures survivor selection.

| Parameter | Type | Default | Description |
| --- | --- | ---: | --- |
| `strategy` | str | --- | `generational`, `truncation`, `steady_state`, `random`, or `stochastic`. |
| `num_replace` | int \\| null | `null` | Number of individuals replaced when used by the selected strategy. |
| `temperature` | float \\| null | `null` | Temperature parameter for strategies that use it. |

## Stopping Criteria

The optional `stopping` block controls early termination.

| Parameter | Type | Default | Description |
| --- | --- | ---: | --- |
| `target_fitness` | float \\| null | `null` | Stop when the target fitness is reached. |
| `minimize` | bool | `true` | If true, lower fitness values are considered better. |
| `patience` | int \\| null | `null` | Stop after this many generations without sufficient improvement. |
| `min_delta` | float | `0.0` | Minimum fitness change considered an improvement. |
| `time_limit_s` | float \\| null | `null` | Wall-clock time limit in seconds. |

## Parallelization

The optional `parallel` block configures parallel fitness evaluation.

| Parameter | Type | Default | Description |
| --- | --- | ---: | --- |
| `backend` | str | `none` | Parallel backend. Currently `none` or `ray`. |
| `num_cpus` | int \\| null | `null` | Number of CPUs allocated to Ray. If omitted, Ray chooses automatically. |
| `address` | str \\| null | `null` | Ray cluster address. If omitted, a local Ray instance is started. |

## Logging

The optional `logging` block controls runtime logging.

| Parameter | Type | Default | Description |
| --- | --- | ---: | --- |
| `lineage` | bool | `false` | Enable detailed per-individual lineage tracking. |

## Modules

The `modules` mapping defines the evolvable parameter representations of
an individual. EvoLib currently provides `vector` and `evonet` modules.

If `type` is omitted, the module is interpreted as a `vector`.

## Vector Module

A vector module stores evolvable numeric parameters.

| Parameter | Type | Default | Description |
| --- | --- | ---: | --- |
| `type` | `"vector"` | `"vector"` | Module type. |
| `dim` | int \\| list\[int\] | --- | Vector length or dimensions for a structured representation. |
| `structure` | str | `flat` | `flat`, `net`, `tensor`, `blocks`, or `grouped`. |
| `initializer` | str | --- | `normal`, `uniform`, `zero`, `fixed`, or `adaptive`. |
| `bounds` | tuple\[float, float\] | `[-1.0, 1.0]` | Hard value bounds. |
| `init_bounds` | tuple\[float, float\] \\| null | `null` | Optional initialization bounds. |
| `shape` | tuple\[int, ...\] \\| null | `null` | Optional explicit shape metadata. |
| `values` | list\[float\] \\| null | `null` | Values used by `initializer: fixed`. |
| `activation` | str \\| null | `null` | Activation used by `structure: net`. |
| `mean` | float \\| null | `0.0` | Mean used by applicable initializers. |
| `std` | float \\| null | `1.0` | Standard deviation used by applicable initializers. |
| `mutation` | dict | --- | Required mutation configuration. |
| `randomize_mutation_strengths` | bool \\| null | `false` | Randomize per-parameter mutation strengths when supported by the strategy. |
| `tau` | float \\| null | `0.0` | Scale factor used by self-adaptive mutation strategies. |
| `crossover` | dict \\| null | `null` | Optional crossover configuration. |

`dim` must contain positive values. With `initializer: fixed`, `values`
is required and `dim` is inferred from `values` when omitted.

For `structure: net`, the initializer must currently be `normal`.

## Mutation Configuration

`MutationConfig` is shared by vector mutation and the main weight
mutation of EvoNet.

### Mutation strategies

  Strategy                   Required fields
  -------------------------- -------------------------------------
  `constant`                 `strength`
  `exponential_decay`        `init_strength`
  `adaptive_global`          `init_strength`, `init_probability`
  `adaptive_individual`      `min_strength`, `max_strength`
  `adaptive_per_parameter`   `min_strength`, `max_strength`

### Mutation fields

| Parameter | Type | Default | Description |
| --- | --- | ---: | --- |
| `strategy` | str | --- | Mutation strategy. |
| `strength` | float \\| null | `null` | Mutation strength used by applicable strategies. |
| `probability` | float \\| null | `null` | Per-parameter mutation probability. |
| `init_strength` | float \\| null | `null` | Initial mutation strength for schedule-based or adaptive strategies. |
| `init_probability` | float \\| null | `null` | Initial mutation probability for schedule-based or adaptive strategies. |
| `min_strength` | float \\| null | `null` | Lower mutation-strength bound. |
| `max_strength` | float \\| null | `null` | Upper mutation-strength bound. |
| `min_probability` | float \\| null | `null` | Lower mutation-probability bound. |
| `max_probability` | float \\| null | `null` | Upper mutation-probability bound. |
| `increase_factor` | float \\| null | `null` | Increase factor used by adaptive strategies. |
| `decrease_factor` | float \\| null | `null` | Decrease factor used by adaptive strategies. |
| `min_diversity_threshold` | float \\| null | `null` | Lower diversity threshold used by adaptive strategies. |
| `max_diversity_threshold` | float \\| null | `null` | Upper diversity threshold used by adaptive strategies. |

Probabilities, when specified, must be in `[0, 1]`. Mutation strengths
must be non-negative.

## Crossover Configuration

Vector and EvoNet modules can contain an optional `crossover` block.

| Parameter | Type | Default | Description |
| --- | --- | ---: | --- |
| `strategy` | str | --- | `none`, `exponential_decay`, `adaptive_global`, or `constant`. |
| `operator` | str \\| null | `null` | `blx`, `arithmetic`, `sbx`, or `intermediate`. |
| `probability` | float \\| null | `null` | Crossover probability. |
| `init_probability` | float \\| null | `null` | Initial probability for adaptive or scheduled strategies. |
| `min_probability` | float \\| null | `null` | Lower probability bound. |
| `max_probability` | float \\| null | `null` | Upper probability bound. |
| `increase_factor` | float \\| null | `null` | Increase factor for adaptive strategies. |
| `decrease_factor` | float \\| null | `null` | Decrease factor for adaptive strategies. |
| `alpha` | float \\| null | `null` | Parameter used by BLX crossover. |
| `eta` | float \\| null | `null` | Parameter used by SBX crossover. |
| `blend_range` | float \\| null | `null` | Range used by intermediate crossover. |

All crossover probabilities, when specified, must be in `[0, 1]`.

## EvoNet Module

EvoNet defines a neural network whose parameters and, optionally,
structure can be evolved.

### Core fields

| Parameter | Type | Default | Description |
| --- | --- | ---: | --- |
| `type` | `"evonet"` | `"evonet"` | Module type. |
| `dim` | list\[int\] | --- | Layer sizes from input to output. |
| `activation` | str \\| list\[str\] | `tanh` | One activation or one activation per layer. |
| `activations_allowed` | list\[str\] \\| null | `null` | Whitelist used for random activation selection. |
| `initializer` | str | `default` | Topology preset: `default`, `unconnected`, or `identity`. |
| `connectivity` | dict | --- | Required connectivity configuration. |
| `weights` | dict | defaults | Weight initialization and bounds. |
| `bias` | dict | defaults | Bias initialization and bounds. |
| `delay` | dict \\| null | `null` | Optional recurrent-delay initialization. |
| `neuron_dynamics` | list\[dict\] \\| null | `null` | Optional neuron dynamics for each layer. |
| `mutation` | dict \\| null | `null` | EvoNet mutation configuration. |
| `crossover` | dict \\| null | `null` | Optional crossover configuration. |

`dim` must contain at least an input and output layer. Layer sizes are
non-negative, allowing empty hidden layers that can later grow through
structural mutation.

When `activation` is a list, its length must match `dim`.

### Connectivity

| Parameter | Type | Default | Description |
| --- | --- | ---: | --- |
| `scope` | str | --- | Feedforward connection scope: `adjacent` or `crosslayer`. |
| `density` | float | --- | Fraction of allowed feedforward connections created initially. Must satisfy `0 < density <= 1`. |
| `recurrent` | list\[str\] | `[]` | Allowed recurrent kinds: `direct`, `lateral`, and/or `indirect`. |

`recurrent: none` is accepted as a YAML shorthand for an empty recurrent
list.

### Weights

| Parameter | Type | Default | Description |
| --- | --- | ---: | --- |
| `initializer` | str \\| null | `null` | `normal`, `uniform`, `zero`, or preset-controlled when omitted. |
| `std` | float \\| null | `null` | Required and greater than 0 for `initializer: normal`. |
| `bounds` | tuple\[float, float\] | `[-1.0, 1.0]` | Mutation and search bounds. |
| `init_bounds` | tuple\[float, float\] \\| null | `null` | Optional initialization bounds inside `bounds`. |

### Bias

| Parameter | Type | Default | Description |
| --- | --- | ---: | --- |
| `initializer` | str \\| null | `null` | `fixed`, `normal`, `uniform`, `zero`, or preset-controlled when omitted. |
| `std` | float \\| null | `null` | Required and greater than 0 for `initializer: normal`. |
| `value` | float \\| null | `null` | Required for `initializer: fixed`. |
| `bounds` | tuple\[float, float\] | `[-0.5, 0.5]` | Mutation and search bounds. |
| `init_bounds` | tuple\[float, float\] \\| null | `null` | Optional initialization bounds inside `bounds`. |

### Delay Initialization

The optional `delay` block initializes delays on recurrent connections.

| Parameter | Type | Default | Description |
| --- | --- | ---: | --- |
| `initializer` | str | `fixed` | `fixed` or `uniform`. |
| `value` | int \\| null | `null` | Fixed delay. Required for `initializer: fixed`. |
| `bounds` | tuple\[int, int\] \\| null | `null` | Inclusive delay range. Required for `initializer: uniform`. |

Delay values must be at least 1.

### Neuron Dynamics

`neuron_dynamics` optionally specifies one dynamics configuration per
layer.

| Parameter | Type | Default | Description |
| --- | --- | ---: | --- |
| `name` | str | `standard` | Neuron dynamics implementation. |
| `params` | dict\[str, float\] | `{}` | Parameters passed to the selected dynamics implementation. |

The number of entries must match the number of layers in `dim`.

## EvoNet Mutation

The main EvoNet `mutation` block uses the common mutation fields for
weights and adds optional overrides for biases, activations, recurrent
delays, and structural mutation.

| Parameter | Type | Default | Description |
| --- | --- | ---: | --- |
| `biases` | dict \\| null | `null` | Optional `MutationConfig` override for biases. |
| `activations` | dict \\| null | `null` | Optional activation mutation configuration. |
| `delay` | dict \\| null | `null` | Optional recurrent-delay mutation configuration. |
| `structural` | dict \\| null | `null` | Optional structural mutation configuration. |

A separate `mutation.weights` block is not supported. The main mutation
fields apply to weights.

### Activation Mutation

| Parameter | Type | Default | Description |
| --- | --- | ---: | --- |
| `probability` | float | --- | Per-neuron mutation probability in `[0, 1]`. |
| `allowed` | list\[str\] \\| null | `null` | Global whitelist for hidden-layer activations. |
| `layers` | dict\[int, list\[str\] \\| `"all"`\] \\| null | `null` | Per-layer activation choices. |

Specify either `allowed` or `layers`, not both.

### Delay Mutation

| Parameter | Type | Default | Description |
| --- | --- | ---: | --- |
| `probability` | float | --- | Per-recurrent-connection mutation probability. |
| `mode` | str | --- | `delta_step` or `resample`. |
| `delta` | int | `1` | Step size for `delta_step`. Must be at least 1. |
| `bounds` | tuple\[int, int\] | --- | Inclusive delay bounds. Minimum delay is 1. |

## Structural Mutation

Structural mutation is configured under `mutation.structural`.

### Add Neuron

| Parameter | Type | Default | Description |
| --- | --- | ---: | --- |
| `probability` | float | `0.0` | Probability of applying the operator. |
| `init_connection_ratio` | float | `1.0` | Fraction of allowed connections added for the new neuron. |
| `init` | str | `zero` | Connection initialization: `none`, `zero`, `near_zero`, or `random`. |
| `activations_allowed` | list\[str\] \\| null | `null` | Activation whitelist for new hidden neurons. |

### Remove Neuron

| Parameter | Type | Default | Description |
| --- | --- | ---: | --- |
| `probability` | float | `0.0` | Probability of applying the operator. |

### Add Connection

| Parameter | Type | Default | Description |
| --- | --- | ---: | --- |
| `probability` | float | `0.0` | Probability of applying the operator. |
| `max` | int | `1` | Maximum number of connections added per event. |
| `init` | str | `zero` | `zero`, `near_zero`, or `random`. |

### Remove Connection

| Parameter | Type | Default | Description |
| --- | --- | ---: | --- |
| `probability` | float | `0.0` | Probability of applying the operator. |
| `max` | int | `1` | Maximum number of connections removed per event. |

### Topology Constraints

| Parameter | Type | Default | Description |
| --- | --- | ---: | --- |
| `recurrent` | list\[str\] | `[]` | Recurrent kinds allowed for structural connection mutation. |
| `connection_scope` | str | `adjacent` | `adjacent` or `crosslayer`. |
| `max_neurons` | int \\| null | `null` | Optional upper bound on the number of neurons. |
| `max_connections` | int \\| null | `null` | Optional upper bound on the number of connections. |

As with initial connectivity, `recurrent: none` is accepted as shorthand
for an empty list.

## Examples

### Minimal Vector Configuration

``` yaml
random_seed: 42
parent_pool_size: 20
offspring_pool_size: 40
max_generations: 100
num_elites: 2

evolution:
  strategy: mu_plus_lambda

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

### EvoNet with HELI and Structural Mutation

``` yaml
parent_pool_size: 20
offspring_pool_size: 40
max_generations: 200
num_elites: 0

evolution:
  strategy: mu_plus_lambda
  heli:
    generations: 10
    offspring_per_seed: 8
    max_fraction: 0.1
    reduce_sigma_factor: 0.5
    seed_selection: fitness

modules:
  brain:
    type: evonet
    dim: [4, 6, 2]
    activation: [linear, tanh, tanh]
    initializer: default

    connectivity:
      scope: crosslayer
      density: 1.0
      recurrent: [direct]

    weights:
      initializer: normal
      std: 0.5
      bounds: [-5.0, 5.0]

    bias:
      initializer: normal
      std: 0.5
      bounds: [-1.0, 1.0]

    delay:
      initializer: uniform
      bounds: [1, 8]

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
        allowed: [tanh, relu, sigmoid]

      delay:
        probability: 0.05
        mode: delta_step
        delta: 1
        bounds: [1, 16]

      structural:
        add_neuron:
          probability: 0.015
          init_connection_ratio: 0.3
          activations_allowed: [tanh]
          init: random

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

### Additional Run Settings and Fixed Vector

``` yaml
random_seed: 1
parent_pool_size: 20
offspring_pool_size: 40
max_generations: 100
num_elites: 1
max_indiv_age: 0

stopping:
  target_fitness: 0.01
  patience: 20
  min_delta: 0.0001
  minimize: true
  time_limit_s: 30.0

selection:
  strategy: tournament
  tournament_size: 3

replacement:
  strategy: generational

parallel:
  backend: none

logging:
  lineage: true

modules:
  main:
    type: vector
    initializer: fixed
    values: [0.0, 1.0, 0.5, -0.5]
    bounds: [-1.0, 1.0]
    mutation:
      strategy: adaptive_individual
      probability: 1.0
      min_strength: 0.01
      max_strength: 0.05
```
