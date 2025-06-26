"""
Example 05-01 (Repair) - Constrained Optimization with Projection.

This version uses a repair strategy instead of penalty functions. After mutation, any
individual that violates the circular constraint is projected back onto the boundary of
the valid region (a circle of radius r).
"""

import matplotlib.pyplot as plt
import numpy as np

from evolib import Indiv, MutationParams, Pop, evolve_mu_lambda

SAVE_FRAMES = True
FRAME_FOLDER = "02_frames_constrained_repair"
CONFIG_FILE = "01_constrained_optimization.yaml"

MAX_RADIUS = 1.5  # constraint: x² + y² ≤ r²


# Repair Mechanism
def repair_to_circle(para: np.ndarray, radius: float = MAX_RADIUS) -> np.ndarray:
    norm = np.linalg.norm(para)
    if norm <= radius:
        return para
    return para * (radius / norm)


# Fitness Function (no penalty needed)
def fitness_function(indiv: Indiv) -> None:
    x, y = indiv.para
    indiv.fitness = (x - 1) ** 2 + (y + 2) ** 2


# Mutation with Repair
def mutation(indiv: Indiv, params: MutationParams) -> None:
    for i in range(len(indiv.para)):
        if np.random.rand() < params.rate:
            indiv.para[i] += np.random.normal(0, params.strength)

    # Apply repair after mutation
    indiv.para = repair_to_circle(indiv.para)


# Initialization
def initialize_population(pop: Pop) -> None:
    for _ in range(pop.parent_pool_size):
        new_indiv = pop.create_indiv()

        # start potentially outside, repair before fitness
        raw = np.random.uniform(-3, 3, size=2)
        new_indiv.para = repair_to_circle(raw)
        pop.add_indiv(new_indiv)

    for indiv in pop.indivs:
        fitness_function(indiv)


# Plotting
def plot_generation(indiv: Indiv, generation: int) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))

    circle = plt.Circle((0, 0), MAX_RADIUS, color="black", fill=False, linestyle="--")
    ax.add_patch(circle)

    x, y = indiv.para
    ax.plot(x, y, "ro", label="Best Solution")

    # Optional: plot constrained theoretical optimum
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
    pop.set_functions(fitness_function, mutation)
    initialize_population(pop)

    for gen in range(pop.max_generations):
        evolve_mu_lambda(pop, fitness_function, mutation)
        pop.sort_by_fitness()
        plot_generation(pop.indivs[0], gen)
        pop.print_status(verbosity=1)


if __name__ == "__main__":
    run_experiment()
