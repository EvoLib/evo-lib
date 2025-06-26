# EvoLib

**EvoLib** is a modular and extensible Python framework for evolutionary algorithms. It supports a wide range of strategies including adaptive mutation, dynamic crossover, and various selection and replacement methods. Its YAML-based configuration and clear structure make it ideal for research, teaching, and experimentation.

## ✨ Features

- ✅ (μ + λ) and (μ, λ) evolution strategies
- ✅ Multiple mutation strategies: constant, exponential decay, adaptive (global/individual/gene-level)
- ✅ Selection methods: tournament, rank-based, roulette, SUS, truncation, Boltzmann
- ✅ Crossover operators: BLX-alpha, arithmetic, SBX, intermediate, heuristic, differential evolution
- ✅ Vector-based and neural network representations
- ✅ Full tracking of fitness and diversity over generations
- ✅ YAML-based configuration for reproducible experiments
- ✅ Easily extensible via plug-in operator loading (`registry.py`)
- ✅ Clean, PEP8-compliant codebase


---


## 📦 Installation

```bash
pip install evolib
```

## 🧬 Example: Function Minimization

```python
from evolib.core.population import Pop
from evolib.strategy import evolve_mu_lambda
from my_custom_module import my_fitness_function, my_mutation_function

pop = Pop("config.yaml")
pop.initialize_random_population()

pop.set_functions(
    fitness_function=my_fitness_function,
    mutation_function=my_mutation_function,
)

for _ in range(pop.max_generations):
    evolve_mu_lambda(pop, my_fitness_function, my_mutation_function)
```

## ⚙️ Configuration Example (`config.yaml`)

```yaml
parent_pool_size: 50
offspring_pool_size: 200
max_generations: 100
representation: VECTOR

mutation:
  strategy: adaptive_individual
  min_rate: 0.01
  max_rate: 0.5
  min_strength: 0.01
  max_strength: 0.3

crossover:
  strategy: blend
  rate: 0.7
```

## 📈 Logging & Visualization

EvoLib logs a range of metrics per generation, including:

- Best, worst, mean, median fitness
- Fitness standard deviation and IQR
- Mutation and crossover parameters
- Diversity metrics (IQR, normalized std, etc.)
These logs can be visualized with tools such as `matplotlib`, `seaborn`, or `pandas`.

## 🔌 Extensibility

New operators can be added under `evolib/operators/` and referenced dynamically:

```python
from evolib.registry import load_strategy

my_selection = load_strategy("selection", "tournament")
```

Valid categories: `"selection"`, `"mutation"`, `"crossover"`, `"replacement"`

## 🧪 Running Tests

```bash
pytest tests/
```

## 📜 License

MIT License — free for academic, personal use.


```{toctree}
:maxdepth: 2
:caption: API Modules

api_population
api_individual
api_mutation
api_selection
api_benchmarks
api_crossover
api_replacement
api_strategy
api_reproduction
api_plotting
api_loss_functions
api_config_loader
api_copy_indiv
api_history_logger
api_registry
api_math_utils
api_config_validator
api_enums
api_structs
api_types
api_numeric
api_utils
```
