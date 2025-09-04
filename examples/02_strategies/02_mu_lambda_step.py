"""
Example 02-02 - Mu Lambda Step

This example demonstrates a basic Mu Plus Lambda and Mu Comma Lambda evolution step.

Didactic only:
    - Operators (`evolve_mu_plus_lambda`, `evolve_mu_comma_lambda`) are called directly
      to reveal their mechanics.
    - In real experiments, strategies are selected via the YAML config and executed
      automatically inside `pop.run()`. Direct calls are not required in practice.

Requirements:
    'population.yaml' must be present in the current working directory
"""

from evolib import (
    Indiv,
    Pop,
    mse_loss,
    simple_quadratic,
)
from evolib.operators.strategy import evolve_mu_comma_lambda, evolve_mu_plus_lambda


# User-defined fitness function
def my_fitness(indiv: Indiv) -> None:
    """Simple fitness function using the quadratic benchmark and MSE loss."""
    expected = 0.0
    predicted = simple_quadratic(indiv.para["test-vector"].vector)
    indiv.fitness = mse_loss(expected, predicted)


def print_population(pop: Pop, title: str) -> None:
    print(f"\n{title}")
    for i, indiv in enumerate(pop.indivs):
        vector = indiv.para["test-vector"].vector
        print(f"  Indiv {i}: Parameter = {vector}, Fitness = {indiv.fitness:.6f}")


# Create and initialize the population.
pop = Pop(config_path="population.yaml", fitness_function=my_fitness)

# Step 1: Evaluate initial fitness
pop.evaluate_fitness()
print_population(pop, "Initial Parents")

# Step 2: Apply Mu Plus Lambda strategy
evolve_mu_plus_lambda(pop)
print_population(pop, "After Mu Plus Lambda")

# Step 3: Apply Mu Comma Lambda strategy
evolve_mu_comma_lambda(pop)
print_population(pop, "After Mu Comma Lambda")
