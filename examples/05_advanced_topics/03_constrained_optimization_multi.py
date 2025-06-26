"""
Example 05-03 (Multi-Constraint) - Constrained Optimization with Multiple Constraints.

This example demonstrates how to apply multiple constraints in evolutionary
optimization. It uses both a circular and a rectangular constraint region and applies
penalties when any constraint is violated.
"""

import matplotlib.pyplot as plt
import numpy as np

from evolib import Indiv, MutationParams, Pop, evolve_mu_lambda

SAVE_FRAMES = True
FRAME_FOLDER = "03_frames_constrained_multi"
CONFIG_FILE = "01_constrained_optimization.yaml"

# Constraints
MAX_RADIUS = 1.5  # Circle constraint: x² + y² ≤ r²
X_MIN, X_MAX = -2.0, 2.0  # Rectangular box: x ∈ [X_MIN, X_MAX]
Y_MIN, Y_MAX = -1.0, 1.0  # y ∈ [Y_MIN, Y_MAX]
PENALTY_FACTOR = 100.0


# Fitness Function
def fitness_function(indiv: Indiv) -> None:
    x, y = indiv.para
    value = (x - 1) ** 2 + (y + 2) ** 2

    penalties = []

    # Circle constraint
    circle_violation = x**2 + y**2 - MAX_RADIUS**2
    if circle_violation > 0:
        penalties.append(circle_violation)

    # Rectangle (box) constraints
    if x < X_MIN:
        penalties.append(X_MIN - x)
    if x > X_MAX:
        penalties.append(x - X_MAX)
    if y < Y_MIN:
        penalties.append(Y_MIN - y)
    if y > Y_MAX:
        penalties.append(y - Y_MAX)

    total_penalty = PENALTY_FACTOR * sum(p**2 for p in penalties)
    indiv.fitness = value + total_penalty


# Mutation
def mutation(indiv: Indiv, params: MutationParams) -> None:
    for i in range(len(indiv.para)):
        if np.random.rand() < params.rate:
            indiv.para[i] += np.random.normal(0, params.strength)


# Initialization
def initialize_population(pop: Pop) -> None:
    for _ in range(pop.parent_pool_size):
        new_indiv = pop.create_indiv()
        new_indiv.para = np.random.uniform(-3, 3, size=2)
        pop.add_indiv(new_indiv)

    for indiv in pop.indivs:
        fitness_function(indiv)


# Plotting
def plot_generation(indiv: Indiv, generation: int) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))

    # Constraint areas
    circle = plt.Circle((0, 0), MAX_RADIUS, color="black", fill=False, linestyle="--")
    ax.add_patch(circle)
    rect = plt.Rectangle(
        (X_MIN, Y_MIN),
        X_MAX - X_MIN,
        Y_MAX - Y_MIN,
        edgecolor="blue",
        facecolor="none",
        linestyle=":",
    )
    ax.add_patch(rect)

    # Points
    x, y = indiv.para
    ax.plot(x, y, "ro", label="Best Solution")

    target = np.array([1.0, -2.0])
    direction = target / np.linalg.norm(target)
    best_on_circle = direction * MAX_RADIUS
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
