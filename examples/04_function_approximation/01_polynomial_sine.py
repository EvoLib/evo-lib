"""
Example 07-01 - Polynomial Approximation of a Target Function (sin(x))

This example demonstrates the use of evolutionary optimization to approximate a
mathematical target function using polynomial regression. Each individual represents the
coefficients of a polynomial. The objective is to minimize the mean squared error
between the target and the approximated function.

Fitness is computed based on the deviation from sin(x) over a fixed range. The best
approximation is plotted at each generation to produce a visual evolution trace (e.g.,
animation).
"""

import matplotlib.pyplot as plt
import numpy as np

from evolib import (
    Indiv,
    MutationParams,
    Pop,
    Strategy,
    adapt_mutation_strengths,
    evolve_mu_lambda,
    mutation_gene_level,
)

# Configuration
TARGET_FUNC = np.sin
x_cheb = np.cos(np.linspace(np.pi, 0, 400))  # [-1, 1]
X_RANGE = (x_cheb + 1) * np.pi  # transformiert nach [0, 2π]
DEGREE = 7
SAVE_FRAMES = True
FRAME_FOLDER = "01_frames_poly"
CONFIG_FILE = "01_polynomial_sine.yaml"


# Fitness Function
def fitness_function(indiv: Indiv) -> None:
    predicted = np.polyval(
        indiv.para[::-1], X_RANGE
    )  # numpy expects highest degree first
    true_vals = TARGET_FUNC(X_RANGE)

    weights = 1.0 + 0.4 * np.abs(np.cos(X_RANGE))
    indiv.fitness = np.average((true_vals - predicted) ** 2, weights=weights)


# Mutation
def mutation(indiv: Indiv, params: MutationParams) -> None:
    # Mutate mutation strengths
    adapt_mutation_strengths(indiv, params)
    # Mutate Parameter
    mutation_gene_level(indiv, params)


# Population Initialization
def initialize_population(pop: Pop, degree: int = DEGREE) -> None:
    for _ in range(pop.parent_pool_size):
        new_indiv = pop.create_indiv()
        # random starting value
        new_indiv.para = np.random.uniform(-0.01, 0.01, degree + 1)

        # Mutationstrength per parameter
        sigma_0 = np.random.uniform(
            pop.min_mutation_strength, pop.max_mutation_strength
        )
        new_indiv.mutation_strengths = [sigma_0 for _ in range(len(new_indiv.para))]

        pop.tau_update_function(new_indiv)  # computes tau based on length of para
        pop.add_indiv(new_indiv)

    for indiv in pop.indivs:
        fitness_function(indiv)


# Plotting per Generation
def plot_approximation(indiv: Indiv, generation: int) -> None:
    y_pred = np.polyval(indiv.para[::-1], X_RANGE)
    y_true = TARGET_FUNC(X_RANGE)

    plt.figure(figsize=(6, 4))
    plt.plot(X_RANGE, y_true, label="Target: sin(x)", color="black")
    plt.plot(X_RANGE, y_pred, label="Best Approx", color="red")
    plt.title(f"Generation {generation}")
    plt.ylim(-1.2, 1.2)
    plt.legend()
    plt.tight_layout()

    if SAVE_FRAMES:
        plt.savefig(f"{FRAME_FOLDER}/gen_{generation:03d}.png")
    plt.close()


# Main
def run_experiment() -> None:
    pop = Pop(CONFIG_FILE)
    pop.set_functions(fitness_function, mutation)
    initialize_population(pop, degree=DEGREE)

    for gen in range(pop.max_generations):
        evolve_mu_lambda(
            pop, fitness_function, mutation, strategy=Strategy.MU_COMMA_LAMBDA
        )
        pop.print_status(verbosity=1)
        pop.sort_by_fitness()
        plot_approximation(pop.indivs[0], gen)


if __name__ == "__main__":
    run_experiment()
