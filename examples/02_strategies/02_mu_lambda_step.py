"""
Example 02-02 - Mu Lambda Step

This example demonstrates a basic Mu Plus Lambda and Mu Comma Lambda evolution step:

Requirements:
- 'population.yaml' and 'individual.yaml' must be present in the
current working directory
"""

from evolib import (
    Indiv,
    Pop,
    Strategy,
    evolve_mu_lambda,
    mse_loss,
    simple_quadratic,
)


# User-defined fitness function
def my_fitness(indiv: Indiv) -> None:
    """Simple fitness function using the quadratic benchmark and MSE loss."""
    expected = 0.0
    predicted = simple_quadratic(indiv.para.vector)
    indiv.fitness = mse_loss(expected, predicted)


def print_population(pop: Pop, title: str) -> None:
    print(f"\n{title}")
    for i, indiv in enumerate(pop.indivs):
        print(
            f"  Indiv {i}: Parameter = {indiv.para.vector}, "
            f"Fitness = {indiv.fitness:.6f}"
        )


# Load configuration and initialize population
my_pop = Pop(config_path="population.yaml")

# Initialize population
my_pop.initialize_population()

# Set fitnessfuction
my_pop.set_functions(fitness_function = my_fitness)

# Evaluate fitness
my_pop.evaluate_fitness()

print_population(my_pop, "Initial Parents")

# Mu Plus Lambda
evolve_mu_lambda(my_pop, strategy=Strategy.MU_PLUS_LAMBDA)

print_population(my_pop, "After Mu Plus Lambda")

# Mu Komma Lambda
evolve_mu_lambda(my_pop, strategy=Strategy.MU_COMMA_LAMBDA)

print_population(my_pop, "After Mu Comma Lambda")
