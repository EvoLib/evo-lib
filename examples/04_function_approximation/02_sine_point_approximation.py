"""
Example 04-02 - Sine Approximation via supportpoints (Y-Vektoren)

Approximates sin(x) by optimizing Y-values at fixed X-support points using evolutionary
strategies. This approach avoids polynomial instability and works with any interpolation
method.
"""

import matplotlib.pyplot as plt
import numpy as np

from evolib import (
    Indiv,
    MutationParams,
    Pop,
    evolve_mu_lambda,
)

# Parameters
NUM_SUPPORT_POINTS = 16
X_SUPPORT = np.linspace(0, 2 * np.pi, NUM_SUPPORT_POINTS)
X_DENSE = np.linspace(0, 2 * np.pi, 400)
Y_TRUE = np.sin(X_DENSE)

SAVE_FRAMES = True
FRAME_FOLDER = "02_frames_point"
CONFIG_FILE = "02_sine_point_approximation.yaml"


# Fitness
def fitness_function(indiv: Indiv) -> None:
    y_support = indiv.para
    y_pred = np.interp(X_DENSE, X_SUPPORT, y_support)
    weights = 1.0 + 0.4 * np.abs(np.cos(X_DENSE))
    indiv.fitness = np.average((Y_TRUE - y_pred) ** 2, weights=weights)


# Mutation
def mutation(indiv: Indiv, params: MutationParams) -> None:
    for i in range(len(indiv.para)):
        if np.random.rand() < params.rate:
            indiv.para[i] += np.random.normal(0, params.strength)


# Initialisierung
def initialize_population(pop: Pop) -> None:
    for _ in range(pop.parent_pool_size):
        new_indiv = pop.create_indiv()
        new_indiv.para = np.random.uniform(-1.0, 1.0, size=len(X_SUPPORT))
        pop.add_indiv(new_indiv)

    for indiv in pop.indivs:
        fitness_function(indiv)


# Visualisierung
def plot_generation(indiv: Indiv, generation: int) -> None:
    y_pred = np.interp(X_DENSE, X_SUPPORT, indiv.para)

    plt.figure(figsize=(6, 4))
    plt.plot(X_DENSE, Y_TRUE, label="Target: sin(x)", color="black")
    plt.plot(X_DENSE, y_pred, label="Best Approx", color="red")
    plt.scatter(X_SUPPORT, indiv.para, color="blue", s=10, label="support points")
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
    initialize_population(pop)

    for gen in range(pop.max_generations):
        evolve_mu_lambda(pop, fitness_function, mutation)
        pop.sort_by_fitness()
        plot_generation(pop.indivs[0], gen)
        pop.print_status(verbosity=1)


if __name__ == "__main__":
    run_experiment()
