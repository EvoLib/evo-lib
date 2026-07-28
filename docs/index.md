# EvoLib

EvoLib is a lightweight and transparent framework for evolutionary computation,
neuroevolution, and small controlled experiments.

The project emphasizes understandable implementations, explicit configuration,
and modular components that can be combined without hiding the evolutionary
process behind a large framework.

## Project areas

- **Evolutionary strategies** — configurable selection, reproduction,
  replacement, crossover, and mutation.
- **Parameter representations** — vectors, network vectors, and EvoNet neural
  networks.
- **Neuroevolution** — weight, bias, activation, recurrent, delay, and
  structural evolution.
- **HELI** — lineage incubation for stabilizing newly mutated network
  structures.
- **Gymnasium integration** — evaluation and visualization on established
  environments.
- **EvoEnv** — small Pygame-based environments for focused evolutionary
  questions, teaching examples, and rapid prototyping.
- **Optional parallel evaluation** — Ray-based fitness evaluation for workloads
  that benefit from parallel execution.

## Start here

New users should begin with the
[getting started guide](getting_started.md), followed by the
[configuration guide](config_guide.md).

The [configuration parameter reference](config_parameter.md) provides a compact
overview of the available YAML settings. The [public API](api_public_api.md)
lists the central imports exposed by `evolib`.

```{toctree}
:maxdepth: 1
:hidden:
:caption: Start here

getting_started
config_guide
config_parameter
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: API – Core

api_core_population
api_core_individual
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: API – Representations

api_representation_vector
api_representation_netvector
api_representation_evonet
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: API – Operators

api_operators_strategy
api_operators_selection
api_operators_replacement
api_operators_reproduction
api_operators_mutation
api_operators_crossover
api_operators_evonet_structural_mutation
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: API – I/O & Utilities

api_utils_loss_functions
api_utils_benchmarks
api_utils_plotting
api_utils_history_logger
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Public API

api_public_api
```
