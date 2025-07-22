"""
Example 07 – Comparison of Selection Strategies.

This example compares different parent selection methods in an evolutionary algorithm.
All configurations use the same optimization task (minimizing the Rastrigin function
towards the zero vector using MSE loss), but differ in how parents are selected.

Included selection methods:
- Tournament
- Rank-Based (linear, exponential)
- Roulette Wheel
- Stochastic Universal Sampling (SUS)
- Boltzmann (Softmax)
- Random
- Truncation

Each method is defined via a separate YAML config file. All runs use the same
offspring pool size (λ) and select 20 parents per generation (num_parents = 20),
creating a moderate selection pressure. This helps highlight differences between
selection strategies in terms of convergence behavior and robustness.

After the runs, best fitness trajectories are plotted to visualize and compare
selection performance.
"""

import numpy as np
import pandas as pd

from evolib import (
    Indiv,
    Pop,
    mse_loss,
    plot_fitness_comparison,
    rastrigin,
)


# Fitness function
def my_fitness(indiv: Indiv) -> None:
    target = np.zeros(indiv.para["test-vector"].dim)
    predicted = rastrigin(indiv.para["test-vector"].vector)
    indiv.fitness = mse_loss(target, predicted)


# Evolution run
def run(config_path: str) -> pd.DataFrame:
    pop = Pop(config_path)
    pop.set_functions(fitness_function=my_fitness)

    for _ in range(pop.max_generations):
        pop.run_one_generation()

    return pop.history_logger.to_dataframe()


# Labels & runs
selection_strategies = {
    "boltzmann": "./07_configs/07_boltzmann.yaml",
    "random": "./07_configs/07_random.yaml",
    "rank_linear": "./07_configs/07_rank_linear.yaml",
    "rank_exponential": "./07_configs/07_rank_exponential.yaml",
    "roulette": "./07_configs/07_roulette.yaml",
    "sus": "./07_configs/07_sus.yaml",
    "tournament": "./07_configs/07_tournament.yaml",
    "truncation": "./07_configs/07_truncation.yaml",
}

runs = {}
for label, config in selection_strategies.items():
    print(f"running: {label}")
    df = run(config)
    runs[label] = df

# Final plot
plot_fitness_comparison(
    histories=list(runs.values()),
    labels=list(runs.keys()),
    metric="best_fitness",
    title="Selection Method Comparison",
    save_path="figures/07_selection_comparison.png",
)
