"""
Example 02-04 — Flexible Strategy Composition.

This example demonstrates how EvoLib supports modular composition of evolutionary
operators using `strategy: flexible`. Instead of relying on a single predefined
strategy such as (mu + lambda), you can configure selection, mutation, crossover,
and replacement explicitly in the YAML file.

This illustrates EvoLib's flexibility: users can either
    - use a predefined evolution strategy for convenience (e.g. `mu_plus_lambda`), or
    - specify `strategy: flexible` to compose their own pipeline.

YAML configuration example (04_flexible.yaml):

```yaml
evolution:
  strategy: flexible

selection:
  strategy: tournament
  tournament_size: 3

replacement:
  strategy: steady_state
  num_replace: 5

modules:
  test-vector:
    type: vector
    dim: 3
    bounds: [-1.0, 1.0]
    initializer: random_vector
    mutation:
      strategy: constant
      strength: 0.01
      probability: 1.0
"""

from evolib import Indiv, Pop, mse_loss, simple_quadratic
from evolib.operators.strategy import evolve_flexible


# User-defined fitness function
def my_fitness(indiv: Indiv) -> None:
    """Simple fitness function using the quadratic benchmark and MSE loss."""
    expected = 0.0
    predicted = simple_quadratic(indiv.para["test-vector"].vector)
    indiv.fitness = mse_loss(expected, predicted)


def print_population(pop: Pop, title: str) -> None:
    print(f"{title}")
    for i, indiv in enumerate(pop.indivs):
        print(
            f"  Indiv {i}: Parameter = {indiv.para['test-vector'].vector}, "
            f"Fitness = {indiv.fitness:.6f}"
        )


# Create and initialize the population from YAML
pop = Pop("04_flexible.yaml", fitness_function=my_fitness)

# Step 1: Evaluate initial fitness
pop.evaluate_fitness()
print_population(pop, "Initial Parents")
print()

# Step 2: Manual loop to illustrate evolve_flexible.
# In normal experiments prefer: pop.run()
for gen in range(pop.max_generations):
    evolve_flexible(pop)
    pop.print_status()

# Step 3: Final inspection - show individuals again for comparison
print_population(pop, "\nFinal Population")
