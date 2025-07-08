"""
Example 04-02 - Sine Approximation via supportpoints (Y-Vektoren)

Approximates sin(x) by optimizing Y-values at fixed X-support points using evolutionary
strategies. This approach avoids polynomial instability and works with any interpolation
method.
"""

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from evolib import Indiv, Pop

# Parameters
X_DENSE = np.linspace(0, 2 * np.pi, 400)
Y_TRUE = np.sin(X_DENSE)

SAVE_FRAMES = True
FRAME_FOLDER = "02_frames_point"
CONFIG_FILE = "02_sine_point_approximation.yaml"


# Fitness
def make_fitness_function(x_support: np.ndarray) -> Callable:
    def fitness_function(indiv: Indiv) -> None:
        y_support = indiv.para.vector
        y_pred = np.interp(X_DENSE, x_support, y_support)
        weights = 1.0 + 0.4 * np.abs(np.cos(X_DENSE))
        indiv.fitness = np.average((Y_TRUE - y_pred) ** 2, weights=weights)

    return fitness_function


# Visualisierung
def plot_generation(indiv: Indiv, generation: int, x_support: np.ndarray) -> None:
    y_pred = np.interp(X_DENSE, x_support, indiv.para.vector)

    plt.figure(figsize=(6, 4))
    plt.plot(X_DENSE, Y_TRUE, label="Target: sin(x)", color="black")
    plt.plot(X_DENSE, y_pred, label="Best Approx", color="red")
    plt.scatter(
        x_support, indiv.para.vector, color="blue", s=10, label="support points"
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

    num_support_points = pop.representation_cfg["dim"]
    x_support = np.linspace(0, 2 * np.pi, num_support_points)

    pop.set_functions(fitness_function=make_fitness_function(x_support))

    for gen in range(pop.max_generations):
        pop.run_one_generation(sort=True)
        plot_generation(pop.best(), gen, x_support)
        pop.print_status(verbosity=1)


if __name__ == "__main__":
    run_experiment()
