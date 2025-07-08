"""
Example 05-05 – Fitness Landscape Exploration with Path.

This variant plots the trajectory of the best individual over a 2D fitness surface. It
visualizes how evolution progresses over generations.
"""

import matplotlib.pyplot as plt
import numpy as np

from evolib import Indiv, Pop, ackley_2d

CONFIG_FILE = "05_fitness_landscape_exploration.yaml"
SAVE_FRAMES = True
FRAME_FOLDER = "05_frames_landscape"

# Grid boundaries for landscape visualization
X_MIN, X_MAX = -32, 32
Y_MIN, Y_MAX = -32, 32


# Objective function
def objective(x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
    return ackley_2d(x, y)


# Fitness evaluation
def fitness_function(indiv: Indiv) -> None:
    x, y = indiv.para.vector
    indiv.fitness = float(objective(x, y))


# Visualization with path
def plot_fitness_landscape_with_path(
    indiv: Indiv, generation: int, path_points: list[tuple[float, float]]
) -> None:
    x_vals = np.linspace(X_MIN, X_MAX, 100)
    y_vals = np.linspace(Y_MIN, Y_MAX, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = objective(X, Y)

    plt.figure(figsize=(6, 5))
    contour = plt.contourf(X, Y, Z, levels=50, cmap="viridis")
    plt.colorbar(contour)

    # Plot path so far
    path = np.array(path_points)
    plt.plot(path[:, 0], path[:, 1], "w--", linewidth=1.0, label="Path")

    # Plot current best
    x, y = indiv.para.vector
    plt.plot(x, y, "ro", label="Best Solution")

    plt.title(f"Fitness Landscape – Gen {generation}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()

    if SAVE_FRAMES:
        plt.savefig(f"{FRAME_FOLDER}/landscape_gen_{generation:03d}.png")
    plt.close()


# Main experiment loop
def run_experiment() -> None:
    pop = Pop(CONFIG_FILE)
    pop.set_functions(fitness_function=fitness_function)

    path_points: list[tuple[float, float]] = []

    for gen in range(pop.max_generations):
        pop.run_one_generation(sort=True)
        best = pop.best()
        x, y = best.para.vector
        path_points.append((x, y))

        plot_fitness_landscape_with_path(best, gen, path_points)
        pop.print_status(verbosity=1)


if __name__ == "__main__":
    run_experiment()
