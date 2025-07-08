"""
Example 05-08 – Vector Control with Obstacle Avoidance (Modern API)

This variant adds circular obstacles to the vector control task. The agent is penalized
for colliding with obstacles while trying to reach the goal.
"""

import matplotlib.pyplot as plt
import numpy as np

from evolib import Indiv, Pop

SAVE_FRAMES = True
FRAME_FOLDER = "08_frames_vector_obstacles"
CONFIG_FILE = "08_vector_control_with_obstacles.yaml"

NUM_STEPS = 8
TARGET = np.array([5.0, 5.0])
START = np.array([0.0, 0.0])
MAX_SPEED = 1.0
PENALTY_FACTOR = 100.0

# List of circular obstacles: (center, radius)
OBSTACLES = [
    (np.array([2.5, 2.5]), 1.0),
    (np.array([4.0, 1.5]), 0.5),
]


# Trajectory simulation with obstacle awareness
def simulate_trajectory(para: np.ndarray) -> np.ndarray:
    pos = START.copy()
    path = [pos.copy()]
    for t in range(NUM_STEPS):
        vx = np.clip(para[t * 2], -MAX_SPEED, MAX_SPEED)
        vy = np.clip(para[t * 2 + 1], -MAX_SPEED, MAX_SPEED)
        pos += np.array([vx, vy])
        path.append(pos.copy())
    return np.array(path)


# Penalty for obstacle collision
def collision_penalty(path: np.ndarray) -> float:
    penalty = 0.0
    for p in path:
        for center, radius in OBSTACLES:
            dist = np.linalg.norm(p - center)
            if dist < radius:
                penalty += float((radius - dist) ** 2)
    return PENALTY_FACTOR * penalty


# Fitness: distance to target + collision penalty
def fitness_function(indiv: Indiv) -> None:
    path = simulate_trajectory(indiv.para.vector)
    final_pos = path[-1]
    dist = np.linalg.norm(final_pos - TARGET)
    penalty = collision_penalty(path)
    indiv.fitness = float(dist + penalty)


# Plot best individual's path with obstacles
def plot_trajectory(indiv: Indiv, generation: int) -> None:
    path = simulate_trajectory(indiv.para.vector)

    plt.figure(figsize=(5, 5))
    plt.plot(path[:, 0], path[:, 1], "o-", color="blue", label="Agent Path")
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


# Main loop
def run_experiment() -> None:
    pop = Pop(CONFIG_FILE)
    pop.set_functions(fitness_function=fitness_function)

    for gen in range(pop.max_generations):
        pop.run_one_generation(sort=True)
        plot_trajectory(pop.best(), gen)
        pop.print_status(verbosity=1)


if __name__ == "__main__":
    run_experiment()
