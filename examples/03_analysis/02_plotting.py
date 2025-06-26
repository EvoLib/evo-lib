"""
Example 03-02 - Plotting

This example shows how to visualize the evolution history collected during a run.
It demonstrates how to:

- Access history data from the population
- Plot fitness statistics over generations
- Interpret trends using matplotlib
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
from evolib.utils.plotting import (
    plot_fitness,
)


def my_fitness(indiv: Indiv) -> None:
    expected = 0.0
    predicted = simple_quadratic(indiv.para)
    indiv.fitness = mse_loss(expected, predicted)


def my_mutation(indiv: Indiv, params: MutationParams) -> None:
    indiv.para = mutate_gauss(indiv.para, params.strength, params.bounds)


def initialize_population(pop: Pop) -> None:
    for _ in range(pop.parent_pool_size):
        new_indiv = pop.create_indiv()
        new_indiv.para = random.uniform(-3, 3)
        pop.add_indiv(new_indiv)
    for indiv in pop.indivs:
        my_fitness(indiv)


# Setup
my_pop = Pop(config_path="population.yaml")
initialize_population(my_pop)

# Evolution
for _ in range(my_pop.max_generations):
    evolve_mu_lambda(my_pop, my_fitness, my_mutation, strategy=Strategy.MU_PLUS_LAMBDA)

# History to DataFrame
history = my_pop.history_logger.to_dataframe()

# Plotting
plot_fitness(history, show=True, save_path="./figures/02_plotting.png")
