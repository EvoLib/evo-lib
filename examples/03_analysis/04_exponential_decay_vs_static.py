"""
Example 04-01 – Exponential Decay of Mutation Rate.

This example demonstrates the impact of exponentially decaying mutation rates
on the performance of a (μ + λ) evolution strategy. It compares a static mutation
rate with an exponentially decreasing one using the Rosenbrock function as the fitness
landscape.

The script runs two experiments with different population configurations and visualizes
the resulting fitness progression over generations.

Visualization:
- A comparison plot of best fitness per generation is saved under:
'./figures/04_exponential_decay.png'
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
    plot_fitness_comparison,
    rosenbrock,
)


# User-defined fitness function
def my_fitness(indiv: Indiv) -> None:
    expected = [1.0, 1.0, 1.0, 1.0]
    predicted = rosenbrock(indiv.para)
    indiv.fitness = mse_loss(expected, predicted)


# User-defined mutation function
def my_mutation(indiv: Indiv, params: MutationParams) -> None:
    for idx, _ in enumerate(indiv.para):
        indiv.para[idx] = mutate_gauss(indiv.para[idx], params.strength, params.bounds)


def initialize_population(pop: Pop) -> None:
    for _ in range(pop.parent_pool_size):
        my_indiv = pop.create_indiv()
        my_indiv.para = [-2.0, 2.0, -1.5, 1.5]  # for better comparison reasons
        pop.add_indiv(my_indiv)
    for indiv in pop.indivs:
        my_fitness(indiv)


def run_experiment(config_path: str) -> pd.DataFrame:
    pop = Pop(config_path)
    initialize_population(pop)

    for _ in range(pop.max_generations):
        evolve_mu_lambda(pop, my_fitness, my_mutation, strategy=Strategy.MU_PLUS_LAMBDA)

    history = pop.history_logger.to_dataframe()
    print(history)

    print(f"Best Indiduum Parameter: {pop.indivs[0].para}")

    return pop.history_logger.to_dataframe()


# Run multiple experiments
history_mutation_constant = run_experiment(config_path="mutation_constant.yaml")
history_mutation_exponential_decay = run_experiment(
    config_path="04_exponential_decay.yaml"
)


# Compare fitness progress
plot_fitness_comparison(
    histories=[history_mutation_constant, history_mutation_exponential_decay],
    labels=["Mutation rate static", "Mutation rate decay"],
    metric="best_fitness",
    title="Best Fitness Comparison (constant vs decay)",
    show=True,
    log=True,
    save_path="./figures/04_exponential_decay.png",
)
