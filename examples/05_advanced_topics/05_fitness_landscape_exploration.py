"""
Example 05-03 - Fitness Landscape Exploration.

This example visualizes the fitness surface of a 2D objective function over a bounded
grid. It helps to understand how evolutionary algorithms navigate the landscape. The
target function is simple but tunable; you can replace it with any benchmark.
"""

import matplotlib.pyplot as plt
import numpy as np

from evolib import (
    Indiv,
    MutationParams,
    Pop,
    ackley_2d,
    evolve_mu_lambda,
)

DIM_X = -32
DIM_Y = 32


# Objective Function
def objective(x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
    return ackley_2d(x, y)


# Fitness Function for population
def fitness_function(indiv: Indiv) -> None:
    x, y = indiv.para
    indiv.fitness = float(objective(x, y))


# Mutation
def mutation(indiv: Indiv, params: MutationParams) -> None:
    for i in range(len(indiv.para)):
        if np.random.rand() < params.rate:
            indiv.para[i] += np.random.normal(0, params.strength)


# Initialization
def initialize_population(pop: Pop) -> None:
    for _ in range(pop.parent_pool_size):
        new_indiv = pop.create_indiv()
        new_indiv.para = [DIM_X, DIM_Y]  # np.random.uniform(DIM_X, DIM_Y, size=2)
        pop.add_indiv(new_indiv)

    for indiv in pop.indivs:
        fitness_function(indiv)


# Landscape Visualization
def plot_fitness_landscape(best_indiv: Indiv, generation: int) -> None:
    x_range = np.linspace(DIM_X, DIM_Y, 100)
    y_range = np.linspace(DIM_X, DIM_Y, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = objective(X, Y)

    plt.figure(figsize=(6, 5))
    contour = plt.contourf(X, Y, Z, levels=50, cmap="viridis")
    plt.colorbar(contour)

    x, y = best_indiv.para
    plt.plot(x, y, "ro", label="Best Solution")
    plt.title(f"Fitness Landscape - Gen {generation}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"05_frames_landscape/landscape_gen_{generation:03d}.png")
    plt.close()


# Main
def run_experiment() -> None:
    CONFIG_FILE = "05_fitness_landscape_exploration.yaml"
    pop = Pop(CONFIG_FILE)
    pop.set_functions(fitness_function, mutation)
    initialize_population(pop)

    for gen in range(pop.max_generations):
        evolve_mu_lambda(pop, fitness_function, mutation)
        pop.sort_by_fitness()
        plot_fitness_landscape(pop.indivs[0], gen)
        pop.print_status(verbosity=1)


if __name__ == "__main__":
    run_experiment()
