"""
Example 05-01 - Constrained Optimization with Evolutionary Strategies.

This example demonstrates how to handle optimization problems with constraints.
Each individual represents a 2D point (x, y), and the goal is to minimize a
cost function under a circular constraint: only solutions inside a radius
of r are valid.

Constraint violation is penalized quadratically.
"""

import matplotlib.pyplot as plt
import numpy as np

from evolib import Indiv, MutationParams, Pop, evolve_mu_lambda

# --- Setup --- #
SAVE_FRAMES = True
FRAME_FOLDER = "01_frames_constrained"
CONFIG_FILE = "01_constrained_optimization.yaml"

# Constraint: x² + y² ≤ r²
MAX_RADIUS = 1.5
PENALTY_FACTOR = 100.0


# --- Fitness Function --- #
def fitness_function(indiv: Indiv) -> None:
    x, y = indiv.para
    value = (x - 1) ** 2 + (y + 2) ** 2  # e.g., distance to target point (1, -2)
    constraint = x**2 + y**2

    if constraint > MAX_RADIUS**2:
        penalty = PENALTY_FACTOR * (constraint - MAX_RADIUS**2)
    else:
        penalty = 0.0

    indiv.fitness = value + penalty


# --- Mutation --- #
def mutation(indiv: Indiv, params: MutationParams) -> None:
    for i in range(len(indiv.para)):
        if np.random.rand() < params.rate:
            indiv.para[i] += np.random.normal(0, params.strength)


# --- Initialization --- #
def initialize_population(pop: Pop) -> None:
    for _ in range(pop.parent_pool_size):
        new_indiv = pop.create_indiv()
        new_indiv.para = np.random.uniform(-3.0, 3.0, size=2)
        pop.add_indiv(new_indiv)

    for indiv in pop.indivs:
        fitness_function(indiv)


# --- Plotting --- #
def plot_generation(indiv: Indiv, generation: int) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))

    # Constraint boundary
    circle = plt.Circle((0, 0), MAX_RADIUS, color="black", fill=False, linestyle="--")
    ax.add_patch(circle)

    # Aktuelles bestes Individuum
    x, y = indiv.para
    ax.plot(x, y, "ro", label="Best Solution")

    # Theorie: bestmöglicher Punkt auf dem Kreis in Richtung Ziel (1, -2)
    target = np.array([1.0, -2.0])
    dir_vec = target / np.linalg.norm(target)
    best_on_circle = dir_vec * MAX_RADIUS
    bx, by = best_on_circle
    ax.plot(bx, by, "go", label="Constrained Optimum")  # grüner Punkt

    # Visuals
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_title(f"Generation {generation}")
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend()

    if SAVE_FRAMES:
        plt.savefig(f"{FRAME_FOLDER}/gen_{generation:03d}.png")
    plt.close()


# --- Main --- #
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
