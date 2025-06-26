"""
Example 05-05 - Rosenbrock Surface with Optimization Path.

Visualizes the 2D Rosenbrock function as a 3D surface. Tracks and displays the
optimization path of the best individual.
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D

from evolib import Indiv, MutationParams, Pop, evolve_mu_lambda, rosenbrock_2d

SAVE_FRAMES = True
FRAME_FOLDER = "06_frames_rosenbrock"
CONFIG_FILE = "06_rosenbrock_surface_path.yaml"

BOUNDS = (-2.0, 2.0)
ZLIM = (0, 2000)
VIEW = dict(elev=35, azim=60)

# Trajektorie
trajectory: list[np.ndarray] = []


# Fitness
def fitness_function(indiv: Indiv) -> None:
    x, y = indiv.para
    indiv.fitness = float(rosenbrock_2d(x, y))


# Mutation
def mutation(indiv: Indiv, params: MutationParams) -> None:
    for i in range(len(indiv.para)):
        if np.random.rand() < params.rate:
            indiv.para[i] += np.random.normal(0, params.strength)


# Initialization
def initialize_population(pop: Pop) -> None:
    for _ in range(pop.parent_pool_size):
        new_indiv = pop.create_indiv()
        new_indiv.para = [-2, -2]  # np.random.uniform(*BOUNDS, size=2)
        pop.add_indiv(new_indiv)
    for indiv in pop.indivs:
        fitness_function(indiv)


def plot_surface_with_path(generation: int, best: Indiv) -> None:
    fig = plt.figure(figsize=(6, 5))
    ax: Axes3D = fig.add_subplot(111, projection="3d")

    # Fitness Surface
    x_range = np.linspace(*BOUNDS, 100)
    y_range = np.linspace(*BOUNDS, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = rosenbrock_2d(X, Y)

    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.7, edgecolor="none")

    # Zielpunkt
    opt_x, opt_y = 1.0, 1.0
    ax.scatter(
        opt_x,
        opt_y,
        rosenbrock_2d(opt_x, opt_y),
        color="green",
        s=60,
        label="Global Optimum",
    )

    # Pfadverlauf
    trajectory.append(best.para.copy())
    if len(trajectory) >= 2:
        path = np.array(trajectory)
        path_z = rosenbrock_2d(path[:, 0], path[:, 1])
        ax.plot3D(path[:, 0], path[:, 1], path_z, "k-", label="Trajectory", lw=1.5)

    # Aktuelles bestes Individuum
    x, y = best.para
    z = rosenbrock_2d(x, y)
    ax.scatter(x, y, z, color="red", s=50, label="Best")

    # Formatierung
    ax.set_xlim(*BOUNDS)
    ax.set_ylim(*BOUNDS)
    ax.set_zlim(*ZLIM)
    ax.set_title(f"Rosenbrock Function - Generation {generation}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")
    ax.view_init(**VIEW)
    ax.legend()

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
        plot_surface_with_path(gen, pop.indivs[0])
        pop.print_status(verbosity=1)


if __name__ == "__main__":
    run_experiment()
