"""
Example 04-03 - Approximation with Noisy Data.

This example investigates the robustness of evolutionary approximation against noisy
target data. It uses fixed support points to approximate sin(x) + normally distributed
noise.
"""

import matplotlib.pyplot as plt
import numpy as np

from evolib import Indiv, MutationParams, Pop, evolve_mu_lambda

# Parameters
NUM_POINTS = 16
X_SUPPORT = np.linspace(0, 2 * np.pi, NUM_POINTS)
X_EVAL = np.linspace(0, 2 * np.pi, 400)
NOISE_STD = 0.1

SAVE_FRAMES = True
FRAME_FOLDER = "03_frames_noise"
CONFIG_FILE = "03_approximation_with_noise.yaml"


# Fitness Function
def fitness_function(indiv: Indiv) -> None:
    y_support = indiv.para
    y_pred = np.interp(X_EVAL, X_SUPPORT, y_support)

    y_true = np.sin(X_EVAL) + np.random.normal(0, NOISE_STD, size=len(X_EVAL))
    indiv.fitness = np.mean((y_true - y_pred) ** 2)


# Mutation
def mutation(indiv: Indiv, params: MutationParams) -> None:
    for i in range(len(indiv.para)):
        if np.random.rand() < params.rate:
            indiv.para[i] += np.random.normal(0, params.strength)


# Initialization
def initialize_population(pop: Pop) -> None:
    for _ in range(pop.parent_pool_size):
        new_indiv = pop.create_indiv()
        new_indiv.para = np.random.uniform(-1.0, 1.0, size=len(X_SUPPORT))
        pop.add_indiv(new_indiv)

    for indiv in pop.indivs:
        fitness_function(indiv)


# Plotting
def plot_generation(indiv: Indiv, generation: int) -> None:
    y_pred = np.interp(X_EVAL, X_SUPPORT, indiv.para)
    y_true = np.sin(X_EVAL)

    plt.figure(figsize=(6, 4))
    plt.plot(X_EVAL, y_true, label="True sin(x)", color="black")
    plt.plot(X_EVAL, y_pred, label="Approximation", color="red")
    plt.scatter(X_SUPPORT, indiv.para, color="blue", s=10, label="Support Points")
    plt.title(f"Gen {generation} - Noisy Fit")
    plt.ylim(-1.5, 1.5)
    plt.legend()
    plt.tight_layout()

    if SAVE_FRAMES:
        plt.savefig(f"{FRAME_FOLDER}/gen_{generation:03d}.png")
    plt.close()


# Main Loop
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
