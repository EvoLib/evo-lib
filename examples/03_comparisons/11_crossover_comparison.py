"""
Example 11 – Comparison of Crossover Operators.

This example compares different crossover operators in an evolutionary algorithm. All
runs approximate the same target vector (e.g. all ones) using MSE loss, but differ in
how offspring are created (BLX, arithmetic, SBX).

Mutation is disabled to isolate the effects of crossover.

After the runs, the best fitness values over time are plotted for comparison.

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
)


# Fitness: distance to target vector
def my_fitness(indiv: Indiv) -> None:
    target = np.ones(indiv.para["test-vector"].dim)
    predicted = indiv.para["test-vector"].vector
    indiv.fitness = mse_loss(target, predicted)


# Evolution run
def run(config_path: str) -> Pop:
    pop = Pop(config_path, fitness_function=my_fitness)
    pop.run()

    return pop


# Crossover operators to compare
crossover_strategies = {
    "blx": "./11_configs/11_blx.yaml",
    "arithmetic": "./11_configs/11_arithmetic.yaml",
    "sbx": "./11_configs/11_sbx.yaml",
    "intermediate": "./11_configs/11_intermediate.yaml",
}

runs = {}
for label, config in crossover_strategies.items():
    print(f"running: {label}")
    runs[label] = run(config)

# Plot results
plot_fitness_comparison(
    histories=list(runs.values()),
    labels=list(runs.keys()),
    metric="best_fitness",
    title="Crossover Operator Comparison",
    save_path="figures/11_crossover_comparison.png",
)
