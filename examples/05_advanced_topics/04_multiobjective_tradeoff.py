"""
Example 05-04 - Multi-Objective Optimization: Fit vs. Smoothness.

This example illustrates multi-objective optimization using scalar logging and
visualizable Pareto analysis. The fitness remains a scalar, but we log both error and
smoothness per individual. This can be used to analyze trade-offs and build Pareto
fronts post-run.
"""

import matplotlib.pyplot as plt
import numpy as np

from evolib import Indiv, MutationParams, Pop, evolve_mu_lambda

SAVE_FRAMES = True
FRAME_FOLDER = "04_frames_multiobjective"
CONFIG_FILE = "04_multiobjective_tradeoff.yaml"

X_DENSE = np.linspace(0, 2 * np.pi, 400)
Y_TRUE = np.sin(X_DENSE)
NUM_POINTS = 16
X_SUPPORT = np.linspace(0, 2 * np.pi, NUM_POINTS)
LAMBDA = 0.1  # optional weighting if scalarized


# Objectives
def compute_mse(y_pred: np.ndarray) -> float:
    return np.mean((Y_TRUE - y_pred) ** 2)


def compute_smoothness(y: np.ndarray) -> float:
    return np.sum(np.diff(y, n=2) ** 2)


# Fitness Function (scalarized, log both)
def fitness_function(indiv: Indiv) -> None:
    y_support = indiv.para
    y_pred = np.interp(X_DENSE, X_SUPPORT, y_support)

    mse = compute_mse(y_pred)
    smooth = compute_smoothness(y_support)

    # scalarized fitness (still single-objective)
    indiv.fitness = mse + LAMBDA * smooth

    # log both for analysis
    indiv.extra_metrics = {"mse": mse, "smoothness": smooth}


# Mutation
def mutation(indiv: Indiv, params: MutationParams) -> None:
    for i in range(len(indiv.para)):
        if np.random.rand() < params.rate:
            indiv.para[i] += np.random.normal(0, params.strength)


# Initialization
def initialize_population(pop: Pop) -> None:
    for _ in range(pop.parent_pool_size):
        new_indiv = pop.create_indiv()
        new_indiv.para = np.random.uniform(-1.0, 1.0, size=NUM_POINTS)
        pop.add_indiv(new_indiv)

    for indiv in pop.indivs:
        fitness_function(indiv)


# Plotting
def plot_generation(indiv: Indiv, generation: int) -> None:
    y_pred = np.interp(X_DENSE, X_SUPPORT, indiv.para)

    plt.figure(figsize=(6, 4))
    plt.plot(X_DENSE, Y_TRUE, label="Target", color="black")
    plt.plot(X_DENSE, y_pred, label="Best Approx", color="red")
    plt.scatter(X_SUPPORT, indiv.para, color="blue", s=10, label="Support Points")
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

    mse_vals = [ind.extra_metrics["mse"] for ind in pop.indivs]
    smooth_vals = [ind.extra_metrics["smoothness"] for ind in pop.indivs]

    plt.scatter(smooth_vals, mse_vals)
    plt.xlabel("Smoothness")
    plt.ylabel("MSE")
    plt.title("Objective Space")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    run_experiment()
