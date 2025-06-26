"""
Example 02-03 - Compare Runs

This example demonstrates how to run the same optimization with different settings
(e.g. mutation strength) and compare their results using the fitness history.
"""

import pandas as pd

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
from evolib.utils.plotting import plot_fitness_comparison


def my_fitness(indiv: Indiv) -> None:
    expected = 0.0
    predicted = simple_quadratic(indiv.para)
    indiv.fitness = mse_loss(expected, predicted)


def my_mutation(indiv: Indiv, params: MutationParams) -> None:
    indiv.para = mutate_gauss(indiv.para, params.strength, params.bounds)


def initialize_population(pop: Pop) -> None:
    for _ in range(pop.parent_pool_size):
        new_indiv = pop.create_indiv()
        new_indiv.para = 3  # for better comparison reasons
        pop.add_indiv(new_indiv)
    for indiv in pop.indivs:
        my_fitness(indiv)


def run_experiment(mutation_strength: float) -> pd.DataFrame:
    pop = Pop(config_path="population.yaml")
    initialize_population(pop)

    pop.mutation_strength = mutation_strength

    for _ in range(pop.max_generations):
        evolve_mu_lambda(pop, my_fitness, my_mutation, strategy=Strategy.MU_PLUS_LAMBDA)

    return pop.history_logger.to_dataframe()


# Run multiple experiments
history_low = run_experiment(mutation_strength=0.1)
history_high = run_experiment(mutation_strength=0.5)

# Compare fitness progress
plot_fitness_comparison(
    histories=[history_low, history_high],
    labels=["Mutation σ = 0.1", "Mutation σ = 0.5"],
    metric="best_fitness",
    title="Best Fitness Comparison (Low vs High Mutation)",
    show=True,
    save_path="./figures/02_Compare_Runs.png",
)
