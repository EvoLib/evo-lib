# EvoLib – A Modular Framework for Evolutionary Computation

[![Docs Status](https://readthedocs.org/projects/evolib/badge/?version=latest)](https://evolib.readthedocs.io/en/latest/)
[![Code Quality & Tests](https://github.com/EvoLib/evo-lib/actions/workflows/ci.yml/badge.svg)](https://github.com/EvoLib/evo-lib/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI version](https://badge.fury.io/py/evolib.svg)](https://pypi.org/project/evolib/)
[![Project Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/EvoLib/evo-lib)

<p align="center">
  <img src="https://raw.githubusercontent.com/EvoLib/evolib/main/assets/evolib_256.png" alt="EvoLib Logo" width="256"/>
</p>

**EvoLib** is a modular and extensible framework for implementing and analyzing evolutionary algorithms in Python.\
It supports classical strategies such as (μ, λ) and (μ + λ) Evolution Strategies, Genetic Algorithms, and Neuroevolution – with a strong focus on clarity, modularity, and didactic value.

---

## 🚀 Key Features

- 🧬 **Configurable Evolution**: Define evolutionary strategies via simple YAML files.
- 🧪 **Modular Design**: Easily swap mutation, selection, and crossover strategies.
- 📈 **Built-in Logging**: Fitness tracking and history recording out-of-the-box.
- 🎓 **Educational Focus**: Clear, didactic examples and extensible code structure.
- 🤖 **Future-Ready**: Neuroevolution and neural representations coming soon.
- ✅ **Type-Checked**: With [mypy](https://mypy-lang.org/) and PEP8 compliance.

### 🧠 Planned: Neural Networks & Neuroevolution

Support for neural network-based individuals and neuroevolution strategies is currently in development.

> ⚠️ **This project is in early development (alpha)**. Interfaces and structure may change.

---

<p align="center">
  <img src="https://raw.githubusercontent.com/EvoLib/evo-lib/main/examples/05_advanced_topics/08_frames_vector_obstacles/08_vector_control_obstacles.gif" alt="Sample Plott" width="512"/>
</p>

---

## 📂 Directory Structure

```
evolib/
├── core/           # Population, Individual
├── operators/      # Crossover, mutation, selection, replacement
├── utils/          # Losses, plotting, config loaders, benchmarks
├── globals/        # Enums and constants
├── config/         # YAML config files
├── examples/       # Educational and benchmark scripts
└── api.py          # Central access point (auto-generated)
```

---

## 📦 Installation

```bash
pip install evolib
```

Requirements: Python 3.9+ and packages in `requirements.txt`.

---

## 🧪 Example Usage

```python
from evolib import Pop
from my_fitness import fitness_function
from my_mutation import mutate_custom

pop = Pop(config_path="config/my_experiment.yaml")
pop.set_functions(fitness_function=fitness_function,
                  mutation_function=mutate_custom)
pop.initialize_population()

for _ in range(pop.max_generations):
    pop.run_one_generation()
```

For full examples, see 📁[`examples/`](https://github.com/EvoLib/evo-lib/tree/main/examples) – including plotting, adaptive mutation, and benchmarking.

---

# ⚙️ Configuration Example (.yaml)

```yaml
parent_pool_size: 10
offspring_pool_size: 40
max_generations: 100
max_indiv_age: 3
num_elites: 1

evolution:
  strategy: mu_plus_lambda

mutation:
  strategy: adaptive_individual
  init_probability: 0.8
  min_probability: 0.2
  max_probability: 1.0
  init_strength: 0.5
  min_strength: 0.01
  max_strength: 1.0

crossover:
  strategy: none
```

---

## 📚 Use Cases

EvoLib is designed to support a wide range of applications, including:

- ✅ **Benchmark optimization**: Solve classic problems like Sphere, Rastrigin, Ackley, etc.
- 🧪 **Hyperparameter tuning**: Use evolutionary strategies to optimize black-box functions.
- 🧬 **Strategy comparison**: Test and evaluate different mutation, selection, and crossover methods.
- 🎓 **Educational use**: Clear API and examples for teaching evolutionary computation concepts.
- 🧠 **Neuroevolution (planned)**: Evolve neural networks and control policies (structure & weights).

---

## 🧠 Roadmap

- [x] Adaptive Mutation (global, individual, per-parameter)
- [x] Flexible Crossover Strategies (BLX, intermediate, none)
- [x] Strategy Comparisons via Examples
- [ ] Neural Network Representations
- [ ] Neuroevolution
- [ ] Visualization Tools for Evolution Progress

---

## 📚 Documentation 

Documentation for EvoLib is available at: 👉 https://evolib.readthedocs.io/en/latest/

---

## 🪪 License

This project is licensed under the [MIT License](https://github.com/EvoLib/evo-lib/tree/main/LICENSE).

---

## 🙏 Acknowledgments

Inspired by classical evolutionary computation techniques and designed for clarity, modularity, and pedagogical use.
