"""
Example 05-06 – Rosenbrock Surface with Optimization Path.

Visualizes the 2D Rosenbrock function as a 3D surface and tracks the path of the best
solution over time.
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D

from evolib import Indiv, Pop, rosenbrock_2d

CONFIG_FILE = "06_rosenbrock_surface_path.yaml"
FRAME_FOLDER = "06_frames_rosenbrock"
SAVE_FRAMES = True

BOUNDS = (-2.0, 2.0)
ZLIM = (0, 2000)
VIEW = dict(elev=35, azim=60)

trajectory: list[np.ndarray] = []  # best-of-generation history


# Fitness
def fitness_function(indiv: Indiv) -> None:
    x, y = indiv.para.vector
    indiv.fitness = float(rosenbrock_2d(x, y))


# Plotting
def plot_surface_with_path(generation: int, best: Indiv) -> None:
    fig = plt.figure(figsize=(6, 5))
    ax: Axes3D = fig.add_subplot(111, projection="3d")

    # Draw surface
    x_range = np.linspace(*BOUNDS, 100)
    y_range = np.linspace(*BOUNDS, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = rosenbrock_2d(X, Y)
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.7, edgecolor="none")

    # Global optimum
    opt_x, opt_y = 1.0, 1.0
    ax.scatter(
        opt_x, opt_y, rosenbrock_2d(opt_x, opt_y), color="green", s=60, label="Optimum"
    )

    # Current best
    x, y = best.para.vector
    z = rosenbrock_2d(x, y)
    ax.scatter(x, y, z, color="red", s=50, label="Best")

    # Draw path so far
    trajectory.append(best.para.vector.copy())
    if len(trajectory) >= 2:
        path = np.array(trajectory)
        z_path = rosenbrock_2d(path[:, 0], path[:, 1])
        ax.plot3D(path[:, 0], path[:, 1], z_path, "k-", label="Path", lw=1.5)

    # Formatting
    ax.set_xlim(*BOUNDS)
    ax.set_ylim(*BOUNDS)
    ax.set_zlim(*ZLIM)
    ax.set_title(f"Rosenbrock Surface – Generation {generation}")
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
    pop.set_functions(fitness_function=fitness_function)

    for gen in range(pop.max_generations):
        pop.run_one_generation(sort=True)
        plot_surface_with_path(gen, pop.best())
        pop.print_status(verbosity=1)


if __name__ == "__main__":
    run_experiment()
