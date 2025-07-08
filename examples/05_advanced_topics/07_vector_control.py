"""
Example 05-07 – Vector-Based Control Task (No Neural Net)

This example shows how a parameter vector directly controls an agent moving through 2D
space. The goal is to reach a target point using a sequence of velocity vectors.
"""

import matplotlib.pyplot as plt
import numpy as np

from evolib import Indiv, Pop

SAVE_FRAMES = True
FRAME_FOLDER = "07_frames_vector_control"
CONFIG_FILE = "07_vector_control.yaml"

NUM_STEPS = 8
START = np.array([0.0, 0.0])
TARGET = np.array([5.0, 5.0])
MAX_SPEED = 1.0


# Simulate trajectory from parameter vector
def simulate_trajectory(para: np.ndarray) -> np.ndarray:
    pos = START.copy()
    for t in range(NUM_STEPS):
        vx = np.clip(para[t * 2 + 0], -MAX_SPEED, MAX_SPEED)
        vy = np.clip(para[t * 2 + 1], -MAX_SPEED, MAX_SPEED)
        pos += np.array([vx, vy])
    return pos


# Fitness function: distance to target
def fitness_function(indiv: Indiv) -> None:
    final_pos = simulate_trajectory(indiv.para.vector)
    indiv.fitness = float(np.linalg.norm(final_pos - TARGET))


# Plot trajectory of best individual
def plot_trajectory(indiv: Indiv, generation: int) -> None:
    pos = START.copy()
    traj = [pos.copy()]
    for t in range(NUM_STEPS):
        vx = np.clip(indiv.para.vector[t * 2 + 0], -MAX_SPEED, MAX_SPEED)
        vy = np.clip(indiv.para.vector[t * 2 + 1], -MAX_SPEED, MAX_SPEED)
        pos += np.array([vx, vy])
        traj.append(pos.copy())

    traj_arr = np.array(traj)

    plt.figure(figsize=(5, 5))
    plt.plot(traj_arr[:, 0], traj_arr[:, 1], "o-", label="Agent Path", color="blue")
    plt.plot(*START, "ks", label="Start")
    plt.plot(*TARGET, "r*", label="Target", markersize=10)
    plt.xlim(-1, 6)
    plt.ylim(-1, 6)
    plt.grid(True)
    plt.title(f"Generation {generation}")
    plt.legend()
    plt.tight_layout()

    if SAVE_FRAMES:
        plt.savefig(f"{FRAME_FOLDER}/gen_{generation:03d}.png")
    plt.close()


# Main
def run_experiment() -> None:
    pop = Pop(CONFIG_FILE)
    pop.set_functions(fitness_function=fitness_function)

    for gen in range(pop.max_generations):
        pop.run_one_generation(sort=True)
        plot_trajectory(pop.best(), gen)
        pop.print_status(verbosity=1)


if __name__ == "__main__":
    run_experiment()
