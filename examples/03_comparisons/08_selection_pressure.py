"""
Example 08 – Selection Pressure via num_parents (fixed strategy)

This script illustrates how selection pressure changes with the number of parents. A
fixed selection strategy (e.g. rank_linear) is used, and only num_parents is varied.

Note:
    Reproducibility is controlled via the `random_seed` field in the YAML config.
    Set it to an integer for deterministic runs or to null/omit it for stochastic runs.
"""

from evolib import (
    Indiv,
    Pop,
    mse_loss,
    plot_fitness_comparison,
    rastrigin,
)


# Fitness function
def my_fitness(indiv: Indiv) -> None:
    target = [0.0, 0.0, 0.0, 0.0]
    predicted = rastrigin(indiv.para["test-vector"].vector)
    indiv.fitness = mse_loss(target, predicted)


def run(config_path: str) -> Pop:
    pop = Pop(config_path, fitness_function=my_fitness)
    pop.run()
    return pop


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
