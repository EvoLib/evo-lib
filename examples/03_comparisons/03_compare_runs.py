"""
Example 02-03 - Compare Runs

This example demonstrates how to run the same optimization with different settings
(e.g. mutation strength) and compare their results using the fitness history.
"""

import pandas as pd

from evolib import Indiv, Pop, mse_loss, simple_quadratic
from evolib.utils.plotting import plot_fitness_comparison


def my_fitness(indiv: Indiv) -> None:
    expected = 0.0
    predicted = simple_quadratic(indiv.para.vector)
    indiv.fitness = mse_loss(expected, predicted)


def run_experiment(mutation_strength: float) -> pd.DataFrame:
    pop = Pop(config_path="population.yaml")
    pop.set_functions(fitness_function=my_fitness)

    for _ in range(pop.max_generations):
        pop.run_one_generation()

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
