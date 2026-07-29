# EvoLib

EvoLib is a framework for evolutionary computation, neuroevolution, and small
controlled experiments.

It uses explicit configuration and modular components. Evolutionary strategies,
parameter representations, operators, and evaluation environments can be
selected and combined separately.

## Project areas

* **Evolutionary strategies** — configurable selection, reproduction,
  replacement, crossover, and mutation.
* **Parameter representations** — vectors, network vectors, and EvoNet neural
  networks.
* **Neuroevolution** — evolution of weights, biases, activation functions,
  recurrent connections, delays, and network structures.
* **HELI** — lineage incubation for newly mutated network structures.
* **Gymnasium integration** — evaluation and visualization using Gymnasium
  environments.
* **EvoEnv** — small Pygame-based environments for controlled evolutionary
  experiments and teaching examples.
* **Parallel evaluation** — optional Ray-based fitness evaluation.

## Start here

New users should begin with the
[getting started guide](getting_started.md), followed by the
[configuration guide](config_guide.md).

The [configuration parameter reference](config_parameter.md) lists the
available YAML settings. Documentation for the included Pygame-based
environments is available under [EvoEnv](evoenv.md).

```{toctree}
:maxdepth: 1
:hidden:
:caption: Guides

getting_started
config_guide
config_parameter
evoenv
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

