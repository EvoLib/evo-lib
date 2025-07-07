"""
Example 2.2 - Mu Lambd

This example demonstrates the basic Mu Plus Lambda and Mu Comma Lambda evolution:
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
    print(f"{title}")
    for i, indiv in enumerate(pop.indivs):
        print(
            f"  Indiv {i}: Parameter = {indiv.para.vector},"
            f"Fitness = {indiv.fitness:.6f}"
        )


# Load configuration and initialize population
pop = Pop(config_path="population.yaml")

# Initialize population
pop.initialize_population()

# Set fitnessfuction
pop.set_functions(fitness_function=my_fitness)

# Evaluate fitness
pop.evaluate_fitness()

print_population(pop, "Initial Parents")

# Mu Plus Lambda
for gen in range(pop.max_generations):
    evolve_mu_lambda(pop, strategy=Strategy.MU_PLUS_LAMBDA)
    pop.print_status()
