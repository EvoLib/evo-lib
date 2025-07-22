"""
Example 08 – Selection Pressure via num_parents (fixed strategy)

This script illustrates how selection pressure changes with the number of parents. A
fixed selection strategy (e.g. rank_linear) is used, and only num_parents is varied.
"""

import random

import numpy as np
import pandas as pd

from evolib import (
    Indiv,
    Pop,
    mse_loss,
    plot_fitness_comparison,
    rastrigin,
)

random.seed(42)
np.random.seed(42)


# Fitness function
def my_fitness(indiv: Indiv) -> None:
    target = np.zeros(indiv.para["test-vector"].dim)
    predicted = rastrigin(indiv.para["test-vector"].vector)
    indiv.fitness = mse_loss(target, predicted)


def run(config_path: str) -> pd.DataFrame:
    pop = Pop(config_path)
    pop.set_functions(fitness_function=my_fitness)

    for _ in range(pop.max_generations):
        pop.run_one_generation()

    return pop.history_logger.to_dataframe()


# Variation over num_parents
config_variants = {
    "num_parents=10": "./08_configs/08_selection_pressure_10.yaml",
    "num_parents=20": "./08_configs/08_selection_pressure_20.yaml",
    "num_parents=40": "./08_configs/08_selection_pressure_40.yaml",
    "num_parents=80": "./08_configs/08_selection_pressure_80.yaml",
}

runs = {}
for label, path in config_variants.items():
    print(f"Running {label}")
    runs[label] = run(path)

# Plot results
plot_fitness_comparison(
    histories=list(runs.values()),
    labels=list(runs.keys()),
    metric="best_fitness",
    title="Impact of num_parents on Selection Pressure (rank_linear)",
    save_path="figures/08_selection_pressure.png",
)
