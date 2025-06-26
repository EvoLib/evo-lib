"""
Example 04-02 – Adaptive Global Mutation.

This example demonstrates the use of an adaptive global mutation strategy within
a (mu + lmbda) evolutionary algorithm framework. The mutation strength is updated
globally based on the population configuration, allowing the mutation process to adapt
over time.

Key Elements:
- The benchmark problem is based on the 4-dimensional Rosenbrock function.
- Fitness is computed as the mean squared error between predicted and expected values.
- Gaussian mutation is applied to each individual’s parameters.
- The experiment compares static vs. adaptive mutation rate strategies.
- Results are visualized using a fitness comparison plot across generations.

Requirements:
- 'population.yaml' and 'individual.yaml' must be present in the working directory.
- Mutation strategy is specified via the configuration file.
- Results are saved to './figures/05_adaptive_global.png'
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


def initialize_population(init_pop: Pop) -> None:
    for _ in range(init_pop.parent_pool_size):
        new_indiv = init_pop.create_indiv()
        new_indiv.para = [-2.0, 2.0, -1.5, 1.5]  # for better comparison reasons
        init_pop.add_indiv(new_indiv)
    for indiv in init_pop.indivs:
        my_fitness(indiv)


def run_experiment(config_path: str) -> pd.DataFrame:
    pop = Pop(config_path)
    initialize_population(pop)

    for _ in range(pop.max_generations):
        evolve_mu_lambda(pop, my_fitness, my_mutation, strategy=Strategy.MU_PLUS_LAMBDA)

    pd.set_option("display.max_rows", None)  # Alle Zeilen anzeigen
    history = pop.history_logger.to_dataframe()
    print(history)

    print(f"Best Indiduum Parameter: {pop.indivs[0].para}")

    return pop.history_logger.to_dataframe()


# Run multiple experiments
history_mutation_constant = run_experiment(config_path="mutation_constant.yaml")
history_mutation_adaptive = run_experiment(config_path="05_adaptive_global.yaml")


# Compare fitness progress
plot_fitness_comparison(
    histories=[history_mutation_constant, history_mutation_adaptive],
    labels=["Mutation rate static", "Mutation rate adaptive"],
    metric="best_fitness",
    title="Best Fitness Comparison (constant vs adaptive)",
    show=True,
    log=True,
    save_path="./figures/05_adaptive_global.png",
)
