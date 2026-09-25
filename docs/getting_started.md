# Getting Started

EvoLib uses YAML files to define evolutionary experiments. The configuration
contains the population and parameter representation, while the fitness function
is defined in Python.

This example optimizes the classic **Sphere function** in five dimensions.

## Step 1: Configuration (`quickstart.yaml`)

```yaml
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
    initializer: normal
    bounds: [-5.0, 5.0]

    mutation:
      strategy: constant
      probability: 1.0
      strength: 0.1
```

The configuration defines a population using a five-dimensional vector and
constant mutation.

For more configuration examples, see the
{doc}`Configuration Guide <config_guide>`. For all available fields and
constraints, see {doc}`Configuration Parameters <config_parameter>`.

## Step 2: The experiment (`quickstart.py`)

```python
from evolib import Indiv, Population, plot_fitness, sphere


def my_fitness(indiv: Indiv) -> None:
    x = indiv.para["main"].vector
    indiv.fitness = sphere(x)


pop = Population("quickstart.yaml", fitness_function=my_fitness)
pop.run(verbosity=1)

plot_fitness(pop, show=True)
```

The fitness function evaluates the `main` parameter module and assigns the
result to the individual's fitness. Lower Sphere values are better, with the
optimum at zero.

## Step 3: Run the experiment

```bash
python quickstart.py
```

A run produces output similar to:

```text
start: strategy=EvolutionStrategy.MU_COMMA_LAMBDA, parents(mu)=20, offspring(lambda)=40, max_gen=50
Population: Gen:   1 Fit: 1.53491476
Population: Gen:   2 Fit: 1.26823752
Population: Gen:   3 Fit: 1.20732208
[...]
```

The fitness history can then be inspected in the generated plot:

![quickstart](/img/quickstart.png "quickstart")
