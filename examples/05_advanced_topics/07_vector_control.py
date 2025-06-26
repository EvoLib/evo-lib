"""
Example 05-07 - Vector-Based Control Task (No Neural Net)

This example demonstrates how a fixed-length parameter vector can directly
control an agent's behavior in a simple 2D movement task.

The goal is to reach a fixed target point using a sequence of velocity vectors.
"""

import matplotlib.pyplot as plt
import numpy as np

from evolib import Indiv, MutationParams, Pop, evolve_mu_lambda

# Constants
SAVE_FRAMES = True
FRAME_FOLDER = "07_frames_vector_control"
CONFIG_FILE = "07_vector_control.yaml"

NUM_STEPS = 8  # number of control steps per episode
TARGET = np.array([5.0, 5.0])  # goal position
START = np.array([0.0, 0.0])
MAX_SPEED = 1.0


# Simulation
def simulate_trajectory(para: np.ndarray) -> np.ndarray:
    """Takes a parameter vector and returns the final position."""
    assert len(para) == NUM_STEPS * 2, "Parameter length mismatch"
    pos = START.copy()
    for t in range(NUM_STEPS):
        vx = np.clip(para[t * 2 + 0], -MAX_SPEED, MAX_SPEED)
        vy = np.clip(para[t * 2 + 1], -MAX_SPEED, MAX_SPEED)
        pos += np.array([vx, vy])
    return pos


# Fitness Function
def fitness_function(indiv: Indiv) -> None:
    final_pos = simulate_trajectory(indiv.para)
    dist = np.linalg.norm(final_pos - TARGET)
    indiv.fitness = float(dist)  # minimize distance


# Mutation
def mutation(indiv: Indiv, params: MutationParams) -> None:
    for i in range(len(indiv.para)):
        if np.random.rand() < params.rate:
            indiv.para[i] += np.random.normal(0, params.strength)


# Initialization
def initialize_population(pop: Pop) -> None:
    for _ in range(pop.parent_pool_size):
        new_indiv = pop.create_indiv()
        new_indiv.para = np.random.uniform(-0.5, 0.5, size=NUM_STEPS * 2)
        pop.add_indiv(new_indiv)

    for indiv in pop.indivs:
        fitness_function(indiv)


# Plotting
def plot_trajectory(indiv: Indiv, generation: int) -> None:
    pos = START.copy()
    traj_list = [pos.copy()]
    for t in range(NUM_STEPS):
        vx = np.clip(indiv.para[t * 2 + 0], -MAX_SPEED, MAX_SPEED)
        vy = np.clip(indiv.para[t * 2 + 1], -MAX_SPEED, MAX_SPEED)
        pos += np.array([vx, vy])
        traj_list.append(pos.copy())
    traj = np.array(traj_list)

    plt.figure(figsize=(5, 5))
    plt.plot(traj[:, 0], traj[:, 1], "o-", color="blue", label="Agent path")
    plt.plot(*START, "ks", label="Start")
    plt.plot(*TARGET, "r*", label="Target", markersize=10)
    plt.xlim(-1, 6)
    plt.ylim(-1, 6)
    plt.grid(True)
    plt.legend()
    plt.title(f"Generation {generation}")
    plt.tight_layout()

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
        plot_trajectory(pop.indivs[0], gen)
        pop.print_status(verbosity=1)


if __name__ == "__main__":
    run_experiment()
