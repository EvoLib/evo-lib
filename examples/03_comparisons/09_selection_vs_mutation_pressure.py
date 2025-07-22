"""
Example 09 – Selection vs. Mutation Pressure.

This example compares the impact of selection pressure (controlled via `num_parents`)
and mutation strength on convergence behavior. It uses a fixed selection strategy
(rank_linear) while varying both selection and mutation parameters.

The experiment uses a fixed initial seed and configuration template. Only the selection
and mutation parameters are varied.

Expected observations:
- Low mutation + low num_parents → fast convergence, but risk of premature convergence
- High mutation + high num_parents → more robust but slower
- Moderate settings yield best balance
"""

# Ensure deterministic behavior
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


# Mapping label to config path
parameter_variants = {
    "parents=10, mutation=0.005": "./09_configs/09_rank_linear_p10_m005.yaml",
    "parents=10, mutation=0.010": "./09_configs/09_rank_linear_p10_m010.yaml",
    "parents=40, mutation=0.005": "./09_configs/09_rank_linear_p40_m005.yaml",
    "parents=40, mutation=0.010": "./09_configs/09_rank_linear_p40_m010.yaml",
}

# Run all variants
runs = {}
for label, path in parameter_variants.items():
    print(f"Running {label}")
    df = run(path)
    runs[label] = df

# Plot results
plot_fitness_comparison(
    histories=list(runs.values()),
    labels=list(runs.keys()),
    metric="best_fitness",
    title="Selection vs. Mutation Pressure (rank_linear)",
    save_path="figures/09_selection_vs_mutation.png",
)
