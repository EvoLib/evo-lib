"""
Example 05-04 – Multi-Objective Optimization: Fit vs. Smoothness.

This example demonstrates a scalarized multi-objective fitness function, balancing data
fit (MSE) and smoothness (second derivative).
"""

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from evolib import Indiv, Pop

SAVE_FRAMES = True
FRAME_FOLDER = "04_frames_multiobjective"
CONFIG_FILE = "04_multiobjective_tradeoff.yaml"

X_EVAL = np.linspace(0, 2 * np.pi, 400)
Y_TRUE = np.sin(X_EVAL)
LAMBDA = 0.1  # smoothness penalty weight


# Objectives
def compute_mse(y_pred: np.ndarray) -> float:
    return np.mean((Y_TRUE - y_pred) ** 2)


def compute_smoothness(y: np.ndarray) -> float:
    return np.sum(np.diff(y, n=2) ** 2)


# Fitness Function (scalarized but logs both)
def make_fitness_function(x_support: np.ndarray) -> Callable:
    def fitness_function(indiv: Indiv) -> None:
        y_support = indiv.para.vector
        y_pred = np.interp(X_EVAL, x_support, y_support)

        mse = compute_mse(y_pred)
        smooth = compute_smoothness(y_support)

        indiv.fitness = mse + LAMBDA * smooth
        indiv.extra_metrics = {"mse": mse, "smoothness": smooth}

    return fitness_function


# Plotting
def plot_generation(indiv: Indiv, generation: int, x_support: np.ndarray) -> None:
    y_pred = np.interp(X_EVAL, x_support, indiv.para.vector)

    plt.figure(figsize=(6, 4))
    plt.plot(X_EVAL, Y_TRUE, label="Target", color="black")
    plt.plot(X_EVAL, y_pred, label="Best Approximation", color="red")
    plt.scatter(
        x_support, indiv.para.vector, color="blue", s=10, label="Support Points"
    )
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

    dim = pop.representation_cfg["dim"]
    x_support = np.linspace(0, 2 * np.pi, dim)

    pop.set_functions(fitness_function=make_fitness_function(x_support))

    for gen in range(pop.max_generations):
        pop.run_one_generation(sort=True)
        plot_generation(pop.best(), gen, x_support)
        pop.print_status(verbosity=1)

    # Objective space visualization (Pareto analysis)
    mse_vals = [ind.extra_metrics["mse"] for ind in pop.indivs]
    smooth_vals = [ind.extra_metrics["smoothness"] for ind in pop.indivs]

    plt.figure()
    plt.scatter(smooth_vals, mse_vals)
    plt.xlabel("Smoothness")
    plt.ylabel("MSE")
    plt.title("Objective Space: Pareto Trade-off")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_experiment()
