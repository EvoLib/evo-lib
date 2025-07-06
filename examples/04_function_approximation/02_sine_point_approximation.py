"""
Example 04-02 - Sine Approximation via supportpoints (Y-Vektoren)

Approximates sin(x) by optimizing Y-values at fixed X-support points using evolutionary
strategies. This approach avoids polynomial instability and works with any interpolation
method.
"""

import numpy as np
import matplotlib.pyplot as plt

from evolib import (
    Pop,
    Indiv,
    evolve_mu_lambda,
)

# Parameters
X_DENSE = np.linspace(0, 2 * np.pi, 400)
Y_TRUE = np.sin(X_DENSE)

SAVE_FRAMES = True
FRAME_FOLDER = "02_frames_point"
CONFIG_FILE = "02_sine_point_approximation.yaml"


# Fitness
def fitness_function(indiv: Indiv) -> None:
    y_support = indiv.para.vector
    y_pred = np.interp(X_DENSE, X_SUPPORT, y_support)
    weights = 1.0 + 0.4 * np.abs(np.cos(X_DENSE))
    indiv.fitness = np.average((Y_TRUE - y_pred) ** 2, weights=weights)


# Visualisierung
def plot_generation(indiv: Indiv, generation: int) -> None:
    y_pred = np.interp(X_DENSE, X_SUPPORT, indiv.para.vector)

    plt.figure(figsize=(6, 4))
    plt.plot(X_DENSE, Y_TRUE, label="Target: sin(x)", color="black")
    plt.plot(X_DENSE, y_pred, label="Best Approx", color="red")
    plt.scatter(X_SUPPORT, indiv.para.vector, color="blue", s=10, label="support points")
    plt.title(f"Generation {generation}")
    plt.ylim(-1.2, 1.2)
    plt.legend()
    plt.tight_layout()

    if SAVE_FRAMES:
        plt.savefig(f"{FRAME_FOLDER}/gen_{generation:03d}.png")
    plt.close()


# Main
def run_experiment() -> None:
    global X_SUPPORT

    pop = Pop(CONFIG_FILE)
    pop.initialize_population()
    pop.set_functions(fitness_function=fitness_function)

    num_support_points = pop.representation_cfg["dim"]
    X_SUPPORT = np.linspace(0, 2 * np.pi, num_support_points)

    for gen in range(pop.max_generations):
        evolve_mu_lambda(pop)
        pop.sort_by_fitness()
        plot_generation(pop.indivs[0], gen)
        pop.print_status(verbosity=1)


if __name__ == "__main__":
    run_experiment()
