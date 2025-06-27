"""
Example 2.2 - Mu Lambd

This example demonstrates the basic Mu Plus Lambda and Mu Comma Lambda evolution:
"""

import random

from evolib import (
    Indiv,
    MutationParams,
    Pop,
    Strategy,
    evolve_mu_lambda,
    mse_loss,
    mutate_gauss,
    simple_quadratic,
)


# User-defined fitness function
def my_fitness(indiv: Indiv) -> None:
    """Simple fitness function using the quadratic benchmark and MSE loss."""
    expected = 0.0
    predicted = simple_quadratic(indiv.para)
    indiv.fitness = mse_loss(expected, predicted)


# User-defined mutation function
def my_mutation(indiv: Indiv, params: MutationParams) -> None:
    """
    Mutates the individual's parameters using Gaussian mutation.

    Args:
        indiv (Indiv): The individual to mutate.
        params (MutationParams): Mutation context object that includes:
            - strength (float): Standard deviation of the Gaussian mutation.
            - bounds (tuple[float, float]): Value limits for the mutation.
            - rate (Optional[float]): (Unused) mutation rate, if applicable.
            - bias (Optional[float]): (Unused) bias mutation strength for
              neural networks.
    """
    indiv.para = mutate_gauss(indiv.para, params.strength, params.bounds)


def print_population(pop: Pop, title: str) -> None:
    print(f"{title}")
    for i, indiv in enumerate(pop.indivs):
        print(
            f"  Indiv {i}: Parameter = {indiv.para:.4f},"
            f"Fitness = {indiv.fitness:.6f}"
        )


# Load configuration and initialize population
my_pop = Pop(config_path="population.yaml")

for _ in range(my_pop.parent_pool_size):
    new_indiv = my_pop.create_indiv()
    new_indiv.para = random.uniform(-3, 3)  # Scalar parameter
    my_pop.add_indiv(new_indiv)


for indiv in my_pop.indivs:
    my_fitness(indiv)

print_population(my_pop, "Initial Parents")

# Mu Plus Lambda
for gen in range(my_pop.max_generations):
    evolve_mu_lambda(my_pop, my_fitness, my_mutation, strategy=Strategy.MU_PLUS_LAMBDA)
    my_pop.print_status()
