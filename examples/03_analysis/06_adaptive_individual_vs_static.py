"""
Example 04-06 – Adaptive Individual Mutation vs. Static Mutation.

This example compares the effectiveness of adaptive mutation at the individual level
with a static mutation strength. Each individual has its own mutation strength and
tau value that adapts over time.

This version produces a single plot with:
- Best fitness (log-scaled Y-axis)
- Mean mutation strength (linear Y-axis, right side)
"""

import random
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evolib import (
    Indiv,
    MutationFunction,
    MutationParams,
    Pop,
    Strategy,
    adapt_mutation_rate,
    default_update_tau,
    evolve_mu_lambda,
    mse_loss,
    mutate_gauss,
    rosenbrock,
)


def my_fitness(indiv: Indiv) -> None:
    expected = [1.0, 1.0, 1.0, 1.0]
    predicted = rosenbrock(indiv.para)
    indiv.fitness = mse_loss(expected, predicted)


def mutation_static(indiv: Indiv, params: MutationParams) -> None:
    for i, val in enumerate(indiv.para):
        indiv.para[i] = mutate_gauss(val, params.strength, params.bounds)


def mutation_adaptive(indiv: Indiv, params: MutationParams) -> None:
    adapt_mutation_rate(indiv, params)

    for i, val in enumerate(indiv.para):
        indiv.para[i] = mutate_gauss(val, indiv.mutation_strength, params.bounds)


def initialize_adaptive(pop: Pop) -> None:
    for _ in range(pop.parent_pool_size):
        new_indiv = pop.create_indiv()
        new_indiv.para = np.array([-2.0, 2.0, -1.5, 1.5])
        new_indiv.mutation_strength = random.uniform(
            pop.min_mutation_strength, pop.max_mutation_strength
        )
        pop.tau_update_function(new_indiv)
        pop.add_indiv(new_indiv)
    for indiv in pop.indivs:
        my_fitness(indiv)


def initialize_static(pop: Pop) -> None:
    for _ in range(pop.parent_pool_size):
        new_indiv = pop.create_indiv()
        new_indiv.para = np.array([-2.0, 2.0, -1.5, 1.5])
        pop.add_indiv(new_indiv)
    for indiv in pop.indivs:
        my_fitness(indiv)


def run_experiment(
    config_path: str, mutation_fn: MutationFunction, init_fn: Callable[[Pop], None]
) -> pd.DataFrame:
    pop = Pop(config_path)
    pop.set_functions(my_fitness, mutation_fn, tau_update_function=default_update_tau)
    init_fn(pop)

    for _ in range(pop.max_generations):
        evolve_mu_lambda(pop, my_fitness, mutation_fn, strategy=Strategy.MU_PLUS_LAMBDA)

    return pop.history_logger.to_dataframe()


def plot_combined_comparison(
    histories: list[pd.DataFrame], labels: list[str], save_path: str
) -> None:

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax2 = ax1.twinx()
    for history, label in zip(histories, labels):
        ax1.plot(
            history["generation"],
            history["best_fitness"],
            label=f"{label} – Best Fitness",
            linestyle="-",
        )
        ax2.plot(
            history["generation"],
            history["mutation_strength_mean"],
            label=f"{label} – Mutation Strength",
            linestyle="--",
        )

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Best Fitness (log scale)")
    ax1.set_yscale("log")
    ax2.set_ylabel("Mean Mutation Strength")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    ax1.set_title("Comparison: Best Fitness and Mean Mutation Strength")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


# Run experiments
history_static = run_experiment(
    "mutation_constant.yaml", mutation_static, initialize_static
)
history_adaptive = run_experiment(
    "06_adaptive_individual.yaml", mutation_adaptive, initialize_adaptive
)

# Combined plot
plot_combined_comparison(
    histories=[history_static, history_adaptive],
    labels=["Static", "Adaptive Individual"],
    save_path="./figures/06_adaptive_individual_vs_static.png",
)
