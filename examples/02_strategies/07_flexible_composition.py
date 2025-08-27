"""
Example 2.7 - Flexible Strategy Composition

This example demonstrates the modular composition of selection, mutation,
crossover, and replacement strategies using `evolve_flexible`.
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
pop = Pop("07_flexible.yaml")

# Set fitness function
pop.set_functions(fitness_function=my_fitness)

# Evaluate initial fitness
pop.evaluate_fitness()
print_population(pop, "Initial Parents")

# Run evolution loop using the flexible strategy
for gen in range(pop.max_generations):
    evolve_flexible(pop)
    pop.print_status()
