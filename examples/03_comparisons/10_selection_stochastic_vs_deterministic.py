"""
Example 10 – Selection: Stochastic vs. Deterministic.

This example contrasts three conceptually distinct selection strategies:

    - roulette       (fully stochastic)
    - tournament     (semi-stochastic, depends on tournament size)
    - truncation     (deterministic, always picks the best)

All other configuration parameters are held constant. The purpose is to illustrate
how the *nature* of a selection strategy (not just its parameters) impacts the
convergence behavior of evolutionary algorithms.

Use this example to understand:
- How randomness affects exploration vs. exploitation
- Why deterministic methods may converge faster, but risk premature convergence
- When stochastic selection helps preserve diversity

This complements:
    - 07_selection_comparison.py  → all strategies side-by-side
    - 08_selection_pressure.py    → fine-grained pressure via num_parents
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

# Ensure reproducibility
np.random.seed(42)
random.seed(42)


def fitness_function(indiv: Indiv) -> None:
    target = np.zeros(indiv.para["test-vector"].dim)
    predicted = rastrigin(indiv.para["test-vector"].vector)
    indiv.fitness = mse_loss(target, predicted)


def run(config_path: str) -> pd.DataFrame:
    pop = Pop(config_path)
    pop.set_functions(fitness_function=fitness_function)

    for _ in range(pop.max_generations):
        pop.run_one_generation()

    return pop.history_logger.to_dataframe()


# Selection variants to compare
config_paths = {
    "roulette": "10_configs/10_selection_roulette.yaml",
    "tournament": "10_configs/10_selection_tournament.yaml",
    "truncation": "10_configs/10_selection_truncation.yaml",
}

# Run and collect results
histories = {}
for label, path in config_paths.items():
    print(f"Running {label} selection...")
    df = run(path)
    histories[label] = df

# Plot fitness comparison
plot_fitness_comparison(
    histories=list(histories.values()),
    labels=list(histories.keys()),
    metric="best_fitness",
    title="Stochastic vs. Deterministic Selection",
    save_path="figures/10_selection_stochastic_vs_deterministic.png",
)
