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

Note:
    Reproducibility is controlled via the `random_seed` field in the YAML config.
    Set it to an integer for deterministic runs or to null/omit it for stochastic runs.
"""

import numpy as np

from evolib import (
    Indiv,
    Pop,
    mse_loss,
    plot_fitness_comparison,
    rastrigin,
)


def my_fitness(indiv: Indiv) -> None:
    target = np.zeros(indiv.para["test-vector"].dim)
    predicted = rastrigin(indiv.para["test-vector"].vector)
    indiv.fitness = mse_loss(target, predicted)


def run(config_path: str) -> Pop:
    pop = Pop(config_path, fitness_function=my_fitness)
    pop.run()
    return pop


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
    runs[label] = run(path)

# Plot results
plot_fitness_comparison(
    histories=list(runs.values()),
    labels=list(runs.keys()),
    metric="best_fitness",
    title="Selection vs. Mutation Pressure (rank_linear)",
    save_path="figures/09_selection_vs_mutation.png",
)
