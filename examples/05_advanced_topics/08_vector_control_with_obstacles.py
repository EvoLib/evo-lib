"""
Example 05-04 (Obstacle Variant) – Vector-Based Control with Obstacle Avoidance.

This version adds circular obstacles the agent must avoid while reaching the target.
Collisions are penalized quadratically, while the primary goal remains minimizing
distance.
"""

import matplotlib.pyplot as plt
import numpy as np

from evolib import Indiv, MutationParams, Pop, evolve_mu_lambda

SAVE_FRAMES = True
FRAME_FOLDER = "08_frames_vector_obstacles"
CONFIG_FILE = "08_vector_control_with_obstacles.yaml"

NUM_STEPS = 8
TARGET = np.array([5.0, 5.0])
START = np.array([0.0, 0.0])
MAX_SPEED = 1.0

# Obstacle(s): list of (center, radius)
OBSTACLES = [
    (np.array([2.5, 2.5]), 1.0),
    (np.array([4.0, 1.5]), 0.5),
]
PENALTY_FACTOR = 100.0


# Simulation
def simulate_trajectory(para: np.ndarray) -> np.ndarray:
    pos = START.copy()
    path = [pos.copy()]
    for t in range(NUM_STEPS):
        vx = np.clip(para[t * 2], -MAX_SPEED, MAX_SPEED)
        vy = np.clip(para[t * 2 + 1], -MAX_SPEED, MAX_SPEED)
        pos += np.array([vx, vy])
        path.append(pos.copy())
    return np.array(path)


# Collision Detection
def collision_penalty(path: np.ndarray) -> float:
    penalty = 0.0
    for p in path:
        for center, radius in OBSTACLES:
            dist = np.linalg.norm(p - center)
            if dist < radius:
                penalty += float((radius - dist) ** 2)
    return PENALTY_FACTOR * penalty


# Fitness Function
def fitness_function(indiv: Indiv) -> None:
    path = simulate_trajectory(indiv.para)
    final_pos = path[-1]
    dist = np.linalg.norm(final_pos - TARGET)
    penalty = collision_penalty(path)
    indiv.fitness = float(dist + penalty)


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
    path = simulate_trajectory(indiv.para)

    plt.figure(figsize=(5, 5))
    plt.plot(path[:, 0], path[:, 1], "o-", color="blue", label="Agent path")
    plt.plot(*START, "ks", label="Start")
    plt.plot(*TARGET, "r*", label="Target", markersize=10)

    for center, radius in OBSTACLES:
        circle = plt.Circle(
            center.tolist(), radius, facecolor="gray", alpha=0.3, edgecolor="black"
        )
        plt.gca().add_patch(circle)

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
