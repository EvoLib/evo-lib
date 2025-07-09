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

from evolib import Indiv, Pop

# Configuration
TARGET_FUNC = np.sin
x_cheb = np.cos(np.linspace(np.pi, 0, 400))  # [-1, 1]
X_RANGE = (x_cheb + 1) * np.pi  # transformiert nach [0, 2π]
SAVE_FRAMES = True
FRAME_FOLDER = "01_frames_poly"
CONFIG_FILE = "01_polynomial_sine.yaml"


# Fitness Function
def fitness_function(indiv: Indiv) -> None:
    predicted = np.polyval(
        indiv.para.vector[::-1], X_RANGE
    )  # numpy expects highest degree first
    true_vals = TARGET_FUNC(X_RANGE)
    weights = 1.0 + 0.4 * np.abs(np.cos(X_RANGE))
    indiv.fitness = np.average((true_vals - predicted) ** 2, weights=weights)


# Plotting per Generation
def plot_approximation(indiv: Indiv, generation: int) -> None:
    y_pred = np.polyval(indiv.para.vector[::-1], X_RANGE)
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
    pop.set_functions(fitness_function=fitness_function)

    for gen in range(pop.max_generations):
        pop.run_one_generation(sort=True)
        pop.print_status(verbosity=1)
        plot_approximation(pop.best(), gen)


if __name__ == "__main__":
    run_experiment()
