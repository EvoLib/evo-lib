# EvoLib – A Modular Framework for Evolutionary Computation

[![Docs Status](https://readthedocs.org/projects/evolib/badge/?version=latest)](https://evolib.readthedocs.io/en/latest/)
[![Code Quality & Tests](https://github.com/EvoLib/evo-lib/actions/workflows/ci.yml/badge.svg)](https://github.com/EvoLib/evo-lib/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI version](https://badge.fury.io/py/evolib.svg)](https://pypi.org/project/evolib/)
[![Project Status: Stable](https://img.shields.io/badge/status-stable-green.svg)](https://github.com/EvoLib/evo-lib)

<p align="center">
  <img src="https://raw.githubusercontent.com/EvoLib/evo-lib/main/assets/evolib_256.png" alt="EvoLib Logo" width="256"/>
</p>

EvoLib is a lightweight and transparent framework for evolutionary computation, focusing on simplicity, modularity, and clarity — aimed at experimentation, teaching, and small-scale research rather than industrial-scale applications.

---

## Key Features

- **Transparent design**: configuration via YAML, type-checked validation, and clear module boundaries.  
- **Modular components**: configurable mutation, selection, crossover, and parameter representations.  
- **Examples**: examples cover basic evolutionary mechanisms, neuroevolution, control tasks, and simulation.  
- **Neuroevolution support**: evolvable neural networks with explicit topology, recurrence, delays, and structural mutation (EvoNet).  
- **Gymnasium integration**: run [Gymnasium](https://gymnasium.farama.org) benchmarks (e.g. CartPole, LunarLander) via a simple wrapper.
- **EvoEnv**: build small, controllable Pygame environments for evolutionary experiments.
- **EvoSim**: lightweight support for persistent evolutionary simulations, with built-in examples for resource competition and competitive coevolution.
- **Parallel evaluation (optional)**: basic support for [Ray](https://www.ray.io/) to speed up fitness evaluations.  
- **HELI (Hierarchical Evolution with Lineage Incubation)**  
  Runs short micro-evolutions ("incubations") for structure-mutated individuals, allowing new topologies to stabilize before rejoining the main population.  
- **Quality checks**: static typing with mypy and automated formatting, linting, and tests.  

---

<p align="center">
  <img src="https://raw.githubusercontent.com/EvoLib/evo-lib/main/examples/05_advanced_topics/04_frames_vector_obstacles/04_vector_control_obstacles.gif" alt="Sample Plot" width="512"/>
</p>

---

## Installation

EvoLib requires Python 3.12 or newer.

```bash
pip install evolib
```

Install optional Ray-based parallel evaluation with:

```bash
pip install "evolib[parallel]"
```


---

## Quick Start

Create `quickstart.yaml`:

```yaml
parent_pool_size: 10
offspring_pool_size: 30
max_generations: 20
num_elites: 1
random_seed: 42

evolution:
  strategy: mu_plus_lambda

modules:
  main:
    type: vector
    dim: 8
    bounds: [-1.0, 1.0]
    initializer: uniform

    mutation:
      strategy: constant
      probability: 1.0
      strength: 0.05
```

Create `run_quickstart.py` in the same directory:

```python
from evolib import Indiv, Pop, plot_fitness, sphere


def fitness(indiv: Indiv) -> None:
    """Evaluate one individual using the Sphere benchmark."""
    vector = indiv.para["main"].vector
    indiv.fitness = sphere(vector)


population = Pop("quickstart.yaml", fitness_function=fitness)

population.run(verbosity=1)
plot_fitness(population, show=True)
```

Run the experiment:

```bash
python run_quickstart.py
```

For more examples, see the [`examples/`](examples/) directory.

---

## Advanced Configuration

EvoLib configurations can combine multiple parameter representations and
fine-grained mutation settings within the same individual. For example:

```yaml
modules:
  controller:
    type: vector
    dim: 8
    initializer: normal
    bounds: [-1.0, 1.0]

    mutation:
      strategy: adaptive_individual
      probability: 1.0
      min_strength: 0.01
      max_strength: 0.1

  brain:
    type: evonet
    dim: [4, 6, 2]
    activation: [linear, tanh, tanh]

    connectivity:
      recurrent: none
      scope: adjacent
      density: 1.0

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
          init_connection_ratio: 0.5
```

---

## Documentation

See the [EvoLib documentation](https://evolib.readthedocs.io/en/latest/)
for configuration details, API documentation, and additional guides.

---

## Archival Record (Zenodo)

EvoLib is archived for long-term reproducibility on Zenodo.

**DOI:** https://doi.org/10.5281/zenodo.17793861

---


## Integrations and Environments

### Gymnasium Integration

EvoLib provides a lightweight wrapper for [Gymnasium](https://gymnasium.farama.org/) environments.
This allows you to evaluate evolutionary agents directly on well-known benchmarks such as **CartPole**, **LunarLander**, or **Pendulum**.

- **Headless evaluation**: returns total episode reward as fitness.
- **Visualization**: render episodes and save them as GIFs.
- **Discrete & continuous action spaces** are both supported.

```python
from evolib import GymEnv

env = GymEnv("CartPole-v1", max_steps=500)
fitness = env.evaluate(indiv)         # run one episode
gif = env.visualize(indiv, gen=10)    # render & save as GIF
```
<p align="center">
  <img src="https://raw.githubusercontent.com/EvoLib/evo-lib/main/examples/08_gym/04_frames/04_lunarlander.gif" alt="LunarLander Evolution" width="512"/>
</p>

[Examples](examples/08_gym)

---

### EvoEnv

### EvoEnv

EvoEnv provides small, controllable Pygame environments for evolutionary
experiments with EvoLib. Environments use episodic evaluation and separate
headless simulation, controller integration, and visualization.

The included tasks cover compact sensor-based control, action timing, navigation,
and experiments with evolvable sensor structure.

<p align="center">
<img src="https://raw.githubusercontent.com/EvoLib/evo-lib/main/examples/09_evoenv/04_collector/collector.gif" alt="EvoEnv Collector example" width="512"/>
</p>

[EvoEnv documentation](evoenv/)  
[Examples](examples/09_evoenv/)

---

### EvoSim

EvoSim provides small, persistent multi-agent simulations for evolutionary
experiments with EvoLib.

Unlike episodic environments, individuals coexist, reproduce, and die in a
continuously changing world. Selection can emerge from resource competition,
survival, reproduction, and interactions between populations.

Current examples include resource competition in **Foraging** and competitive
coevolution in **Predator-Prey**.

<p align="center">
  <img src="https://raw.githubusercontent.com/EvoLib/evo-lib/main/examples/10_evosim/02_predator_prey/predator_prey.gif" alt="Predator-Prey sample" width="512"/>
</p>

[EvoSim documentation](evosim/)  
[Examples](examples/10_evosim/)

---

## Learn EvoLib in 5 Steps

EvoLib includes a small set of examples that illustrate the core concepts step by step:

1. [Hello Evolution](examples/01_basic_usage/04_fitness.py) – minimal run with a custom fitness function and visible improvement over generations.
2. [Strategies in Action](examples/02_strategies/03_mu_lambda.py) – (μ + λ) evolution step by step.
3. [Function Approximation](examples/04_function_approximation/02_sine_point_approximation.py) – evolve support points to match a sine curve.
4. [Evolution as Control](examples/05_advanced_topics/04_vector_control_with_obstacles.py) – evolve a controller in an environment.
5. [Neuroevolution with Structural Growth](examples/07_evonet/06_structural_xor.py) – evolve networks with growing topology.

For deeper exploration, see the [full examples directory](examples/)

---

## Acknowledgement

ChatGPT (OpenAI) was used to support documentation, docstrings, language editing, and code refactoring.

---

## License

MIT License – see [MIT License](https://github.com/EvoLib/evo-lib/tree/main/LICENSE).
