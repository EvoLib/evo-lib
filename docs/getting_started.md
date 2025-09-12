# Getting Started

## What is EvoLib?

EvoLib is a lightweight framework for evolutionary computation in Python.
It is designed for **clarity, modularity, and experimentation**, making it well-suited for teaching, small-scale research, and exploring new ideas.

---

## Philosophy

EvoLib follows three guiding principles:

* **Simplicity** – focus on clear, minimal components that are easy to understand.
* **Modularity** – mutation, selection, crossover, and parameter types can be freely combined.
* **Transparency** – configurations are explicit in YAML, and results are easy to inspect.

This makes EvoLib a tool for *learning by doing and quick experimentation*.

---

## Quickstart Example

This minimal setup optimizes the classic **Sphere function** in 5 dimensions.
The configuration is defined in YAML, while the fitness function is provided in Python.

### Step 1: Configuration (`quickstart.yaml`)

```yaml
# Minimal config for Sphere optimization
parent_pool_size: 20
offspring_pool_size: 40
max_generations: 50
num_elites: 0

evolution:
  strategy: mu_comma_lambda

modules:
  main:
    type: vector
    dim: 5
    initializer: normal_vector
    bounds: [-5.0, 5.0]
    mutation:
      strategy: constant
      probability: 1.0
      strength: 0.1
```

#### Explanation of key parameters:

- parent_pool_size / offspring_pool_size: number of parents selected and offspring created per generation.
- max_generations: stop condition based on the total number of generations.
- num_elites: number of top individuals carried over unchanged.
- bounds: lower and upper limits for the vector values.

> ℹ️ See the {doc}`Configuration Guide <config_guide>`.


### Step 2: The experiment (`quickstart.py`)

```python
import numpy as np
from evolib import Indiv, Population, sphere, plot_fitness

# Define fitness function
def my_fitness(indiv: Indiv) -> None:
    x = indiv.para["main"].vector
    indiv.fitness = sphere(x)

# Load population from YAML config
pop = Population("quickstart.yaml", fitness_function=my_fitness)

# Run optimization
pop.run(verbosity=1)

# Plot fitness progress (best, mean, median over generations)
plot_fitness(pop, show=True)
```

### Step 3: Run the experiment (`quickstart.py`)
```bash
python quickstart.py
```

#### Console output:
```text
start: strategy=EvolutionStrategy.MU_COMMA_LAMBDA, parents(mu)=20, offspring(lambda)=40, max_gen=50
Population: Gen:   1 Fit: 1.53491476
Population: Gen:   2 Fit: 1.26823752
Population: Gen:   3 Fit: 1.20732208
Population: Gen:   4 Fit: 1.02901099
Population: Gen:   5 Fit: 0.80762856
Population: Gen:   6 Fit: 0.66001636
Population: Gen:   7 Fit: 0.49777490
[...]
```

#### Plot:
![quickstart](/img/quickstart.png "quickstart")
