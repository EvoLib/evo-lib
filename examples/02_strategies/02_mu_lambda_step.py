"""
Example 02-02 - Mu Lambda Step

This example demonstrates a basic Mu Plus Lambda and Mu Comma Lambda evolution step:

Requirements:
    'population.yaml' must be present in the current working directory
"""

from evolib import (
    Indiv,
    Pop,
    evolve_mu_comma_lambda,
    evolve_mu_plus_lambda,
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


# Automatically creates and initializes the population.
# Note: Use initialize=False to delay or customize population setup.
pop = Pop(config_path="population.yaml")


# Initialize population
# pop.initialize_population()

# Set fitnessfuction
pop.set_functions(fitness_function=my_fitness)

# Evaluate fitness
pop.evaluate_fitness()

print_population(pop, "Initial Parents")

# Mu Plus Lambda
evolve_mu_plus_lambda(pop)

print_population(pop, "After Mu Plus Lambda")

# Mu Komma Lambda
evolve_mu_comma_lambda(pop)

print_population(pop, "After Mu Comma Lambda")
