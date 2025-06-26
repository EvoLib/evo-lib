"""
Example 3.0 - History

This example demonstrates how to log and inspect fitness statistics during evolution.

It introduces:
- Logging of per-generation statistics (e.g., best, mean, std)
- Accessing and printing the history as a DataFrame

Requirements:
- 'population.yaml' and 'individual.yaml' must be present in the
current working directory
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


def my_fitness(indiv: Indiv) -> None:
    """Simple fitness function using the quadratic benchmark and MSE loss."""
    expected = 0.0
    predicted = simple_quadratic(indiv.para)
    indiv.fitness = mse_loss(expected, predicted)


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
    for i, print_indiv in enumerate(pop.indivs):
        print(
            f"  Indiv {i}: Parameter = {indiv.para:.4f},"
            "Fitness = {indiv.fitness:.6f}"
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

history = my_pop.history_logger.to_dataframe()
print(
    history[
        [
            "generation",
            "best_fitness",
            "worst_fitness",
            "mean_fitness",
            "std_fitness",
            "iqr_fitness",
        ]
    ]
)

print("\nFinal History Snapshot (last 5 generations):")
print(
    history[
        [
            "generation",
            "best_fitness",
            "worst_fitness",
            "mean_fitness",
            "std_fitness",
            "iqr_fitness",
        ]
    ].tail()
)

best_overall = history["best_fitness"].min()
print(f"\nBest fitness achieved: {best_overall:.6f}")
