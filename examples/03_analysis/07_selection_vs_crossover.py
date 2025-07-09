"""
Example 05-07 – Selection vs. Crossover.

Compares selection method (Tournament vs. Roulette) and crossover (None vs. BLX-alpha)
in a (μ + λ) Evolution Strategy using constant mutation strength.

Fitness: deviation from target vector using Rosenbrock + MSE loss.
"""

from typing import List

import numpy as np
import pandas as pd

from evolib import (
    Indiv,
    Pop,
    crossover_blend_alpha,
    generate_cloned_offspring,
    mse_loss,
    plot_fitness_comparison,
    replace_mu_lambda,
    rosenbrock,
    selection_roulette,
    selection_tournament,
)

DIM = 100
CONFIG = "07_selection_vs_crossover_static.yaml"


# Fitness function
def my_fitness(indiv: Indiv) -> None:
    target = np.ones(DIM)
    predicted = rosenbrock(indiv.para.vector)
    indiv.fitness = mse_loss(target, predicted)


# Optional crossover (BLX-alpha)
def my_crossover(offspring: List[Indiv]) -> None:
    for i in range(0, len(offspring) - 1, 2):
        p1, p2 = offspring[i], offspring[i + 1]
        c1_vec, c2_vec = crossover_blend_alpha(p1.para.vector, p2.para.vector)
        p1.para.vector = c1_vec
        p2.para.vector = c2_vec


# Initialization using Pop API
def initialize(pop: Pop) -> None:
    pop.initialize_population()
    for indiv in pop.indivs:
        my_fitness(indiv)


# Evolution run
def run(pop: Pop, *, use_crossover: bool, selection_method: str) -> pd.DataFrame:
    for _ in range(pop.max_generations):
        # SELECTION
        if selection_method == "tournament":
            parents = selection_tournament(pop, pop.parent_pool_size, tournament_size=3)
        elif selection_method == "roulette":
            parents = selection_roulette(pop, pop.parent_pool_size)
        else:
            raise ValueError("Unknown selection method")

        # REPRODUCTION
        offspring = generate_cloned_offspring(parents, pop.offspring_pool_size)

        # CROSSOVER
        if use_crossover:
            my_crossover(offspring)

        # MUTATION (via ParaVector)
        for indiv in offspring:
            indiv.mutate()

        # FITNESS
        for indiv in offspring:
            my_fitness(indiv)

        # REPLACEMENT
        replace_mu_lambda(pop, offspring)
        pop.update_statistics()

    return pop.history_logger.to_dataframe()


# Labels & runs
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

    pop = Pop(CONFIG, initialize=False)
    initialize(pop)
    df = run(pop, use_crossover=use_crossover, selection_method=selection_type)
    runs[label] = df

# Final plot
plot_fitness_comparison(
    histories=list(runs.values()),
    labels=list(runs.keys()),
    metric="best_fitness",
    title="Selection and Crossover Comparison",
    save_path="figures/07_crossover_vs_selection.png",
)
