"""
07_crossover_vs_selection.py.

Compares the impact of selection (Tournament vs. Roulette) and crossover
in a (μ + λ) Evolution Strategy using constant mutation strength.

Fitness function: deviation from [1.0, 1.0, 1.0, 1.0] using the Rosenbrock function
and MSE loss.

Four configurations are compared:
1. No crossover + Tournament selection
2. No crossover + Roulette selection
3. BLX-Alpha crossover + Tournament selection
4. BLX-Alpha crossover + Roulette selection
"""

from typing import List

import numpy as np
import pandas as pd

from evolib import (
    Indiv,
    MutationParams,
    Pop,
    create_offspring_mu_lambda,
    crossover_blend_alpha,
    mse_loss,
    mutate_offspring,
    plot_fitness_comparison,
    replace_mu_lambda,
    rosenbrock,
    selection_roulette,
    selection_tournament,
)

BOUNDS = (-2.0, 2.0)
DIM = 100


def my_fitness(indiv: Indiv) -> None:
    # Assigns fitness based on Rosenbrock deviation from [1.0, 1.0, 1.0, 1.0]
    expected = [1.0] * DIM
    predicted = rosenbrock(indiv.para)
    indiv.fitness = mse_loss(expected, predicted)


def mutation_function(indiv: Indiv, params: MutationParams) -> None:
    # Simple Gaussian mutation.
    noise = np.random.normal(0, params.strength, size=len(indiv.para))
    indiv.para += noise
    indiv.para = np.clip(indiv.para, params.bounds[0], params.bounds[1])


def my_crossover(offspring: List[Indiv]) -> None:

    for i in range(0, len(offspring) - 1, 2):
        p1, p2 = offspring[i], offspring[i + 1]
        child1_para, child2_para = crossover_blend_alpha(
            np.array(p1.para), np.array(p2.para)
        )
        offspring[i].para = child1_para
        offspring[i + 1].para = child2_para


def initialize_indivs(pop: Pop, dim: int = DIM) -> None:
    # Initializes population with uniformly random individuals.
    bounds = (-2.0, 2.0)
    for _ in range(pop.parent_pool_size):
        indiv = pop.create_indiv()
        indiv.para = np.random.uniform(bounds[0], bounds[1], size=dim)
        my_fitness(indiv)
        pop.add_indiv(indiv)


def run(pop: Pop, *, use_crossover: bool, selection_method: str) -> pd.DataFrame:
    # Runs the evolution process for a single configuration.

    for _ in range(pop.max_generations):
        # SELECTION
        if selection_method == "tournament":
            parents = selection_tournament(pop, pop.parent_pool_size, tournament_size=3)
        elif selection_method == "roulette":
            parents = selection_roulette(pop, pop.parent_pool_size)
        else:
            raise ValueError("Unknown selection method")

        # REPRODUCTION
        offspring = create_offspring_mu_lambda(parents, pop.offspring_pool_size)

        # OPTIONAL CROSSOVER
        if use_crossover:
            my_crossover(offspring)

        # MUTATION
        mutate_offspring(pop, offspring, mutation_function, bounds=(-2, 2))

        # FITNESS EVALUATION
        for indiv in offspring:
            my_fitness(indiv)

        # REPLACEMENT (μ + λ)
        replace_mu_lambda(pop, offspring)

        pop.update_statistics()

    return pop.history_logger.to_dataframe()


labels = [
    "no_crossover_tournament",
    "no_crossover_roulette",
    "crossover_tournament",
    "crossover_roulette",
]

runs = {}
for label in labels:
    selection_type = "tournament" if "tournament" in label else "roulette"
    use_crossover = "crossover" in label

    pop = Pop("07_selection_vs_crossover_static.yaml")

    initialize_indivs(pop)

    df = run(pop, use_crossover=use_crossover, selection_method=selection_type)
    runs[label] = df

plot_fitness_comparison(
    histories=list(runs.values()),
    labels=list(runs.keys()),
    metric="best_fitness",
    title="Selection and Crossover Comparison",
    save_path="figures/07_crossover_vs_selection.png",
)
