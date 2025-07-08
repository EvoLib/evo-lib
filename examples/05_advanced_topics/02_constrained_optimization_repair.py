"""
Example 05-02 - Constrained Optimization with Repair Strategy.

This variant uses a geometric repair mechanism instead of penalty terms.
After mutation, any solution outside the constraint circle is projected
back onto the boundary.
"""

import matplotlib.pyplot as plt
import numpy as np

from evolib import Indiv, Pop

SAVE_FRAMES = True
FRAME_FOLDER = "02_frames_constrained_repair"
CONFIG_FILE = "01_constrained_optimization.yaml"

# Constraint: x² + y² ≤ r²
MAX_RADIUS = 1.5


# Repair Mechanism: Project back onto constraint circle if needed
def repair_to_circle(vec: np.ndarray, radius: float = MAX_RADIUS) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm <= radius:
        return vec
    return vec * (radius / norm)


# Fitness Function (no penalty, assumes repaired vector)
def fitness_function(indiv: Indiv) -> None:
    indiv.para.vector = repair_to_circle(indiv.para.vector)  # apply repair
    x, y = indiv.para.vector
    indiv.fitness = (x - 1) ** 2 + (y + 2) ** 2


# Plotting
def plot_generation(indiv: Indiv, generation: int) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))

    # Constraint boundary
    circle = plt.Circle((0, 0), MAX_RADIUS, color="black", fill=False, linestyle="--")
    ax.add_patch(circle)

    x, y = indiv.para.vector
    ax.plot(x, y, "ro", label="Best Solution")

    # Theoretical optimum on boundary (direction to target)
    target = np.array([1.0, -2.0])
    best_on_circle = target / np.linalg.norm(target) * MAX_RADIUS
    ax.plot(*best_on_circle, "go", label="Constrained Optimum")

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title(f"Generation {generation}")
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
        plot_generation(pop.indivs[0], gen)
        pop.print_status(verbosity=1)


if __name__ == "__main__":
    run_experiment()
