# EvoLib – A Modular Toolkit for Evolutionary Computation


<p align="center">
  <img src="https://github.com/EvoLib/evolib/blob/main/assets/evolib_256.png" alt="EvoLib Logo" width="256"/>
</p>


**EvoLib** is a modular and extensible framework for implementing and analyzing evolutionary algorithms in Python.\
It supports classical strategies such as (μ, λ) and (μ + λ) Evolution Strategies, Genetic Algorithms, and Neuroevolution – with a strong focus on clarity, modularity, and didactic value.

---

## 🚀 Features

- Individual- and population-level adaptive mutation strategies
- Modular selection methods: tournament, rank-based, roulette, SUS, truncation, Boltzmann
- Multiple crossover operators: heuristic, arithmetic, differential, SBX, etc.
- Configurable via YAML: clean separation of individual and population setups
- Benchmark functions: Sphere, Rosenbrock, Rastrigin, Ackley, Griewank, etc.
- Built-in loss functions (MSE, MAE, Huber, Cross-Entropy)
- Plotting utilities for fitness trends, mutation tracking, diversity
- Designed for extensibility: clean core/operator/utils split
- Sphinx-based documentation (coming soon)

---

<p align="center">
  <img src="https://github.com/EvoLib/evo-lib/blob/main/examples/05_advanced_topics/08_frames_vector_obstacles/08_vector_control_obstacles.gif" alt="Sample Plott" width="512"/>
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

## 🧪 Quickstart Example

```python
from evolib import Pop, Indiv, evolve_mu_lambda, mse_loss, sphere

def fitness(indiv: Indiv) -> None:
    indiv.fitness = mse_loss(0.0, sphere(indiv.para))

pop = Pop(config_path="config/population.yaml")
for _ in range(pop.max_generations):
    evolve_mu_lambda(pop, fitness)
    print(pop)
```

For full examples, see 📁[`examples/`](./examples/) – including plotting, adaptive mutation, and benchmarking.

---

## 📚 Use Cases

- Evolutionary benchmark optimization
- Parameter tuning
- Algorithm comparisons
- Teaching material for evolutionary computation
- Neuroevolution

---

## 🪪 License

This project is licensed under the [MIT License](./LICENSE).

---

## 🙏 Acknowledgments

Inspired by classical evolutionary computation techniques and designed for clarity, modularity, and pedagogical use.

