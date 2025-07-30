"""
Example 04-03 - Approximation with Noisy Data.

This example investigates the robustness of evolutionary approximation against noisy
target data. It uses fixed support points to approximate sin(x) + normally distributed
noise.
"""

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from evolib import Indiv, Pop

# Parameters
X_EVAL = np.linspace(0, 2 * np.pi, 400)
NOISE_STD = 0.1

SAVE_FRAMES = True
FRAME_FOLDER = "03_frames_noise"
CONFIG_FILE = "03_approximation_with_noise.yaml"


# Noisy target function
def get_noisy_target(x: np.ndarray, noise_std: float = NOISE_STD) -> np.ndarray:
    return np.sin(x) + np.random.normal(0, noise_std, size=len(x))


# Fitness Function
def make_fitness_function(x_support: np.ndarray) -> Callable:
    def fitness_function(indiv: Indiv) -> None:
        y_support = indiv.para["points"].vector
        y_pred = np.interp(X_EVAL, x_support, y_support)
        y_target = get_noisy_target(X_EVAL)
        indiv.fitness = np.mean((y_target - y_pred) ** 2)

    return fitness_function


# Plotting Function
def plot_generation(
    indiv: Indiv,
    generation: int,
    x_support: np.ndarray,
    show_noisy_target: bool = True,
) -> None:
    y_pred = np.interp(X_EVAL, x_support, indiv.para["points"].vector)
    y_true = np.sin(X_EVAL)

    plt.figure(figsize=(6, 4))
    plt.plot(X_EVAL, y_true, label="sin(x)", color="black")

    if show_noisy_target:
        y_noisy = get_noisy_target(X_EVAL)
        plt.plot(X_EVAL, y_noisy, label="Noisy target", color="gray", linestyle=":")

    plt.plot(X_EVAL, y_pred, label="Approximation", color="red")
    plt.scatter(
        x_support,
        indiv.para["points"].vector,
        color="blue",
        s=10,
        label="Support Points",
    )

    plt.title(f"Gen {generation} – Robust Fit")
    plt.ylim(-1.5, 1.5)
    plt.legend()
    plt.tight_layout()

    if SAVE_FRAMES:
        plt.savefig(f"{FRAME_FOLDER}/gen_{generation:03d}.png")
    plt.close()


# Main Execution
def run_experiment() -> None:
    pop = Pop(CONFIG_FILE)

    dim = pop.sample_indiv.para["points"].dim
    x_support = np.linspace(0, 2 * np.pi, dim)

    pop.set_functions(fitness_function=make_fitness_function(x_support))

    for gen in range(pop.max_generations):
        pop.run_one_generation(sort=True)
        plot_generation(pop.indivs[0], gen, x_support, show_noisy_target=True)
        pop.print_status(verbosity=1)


if __name__ == "__main__":
    run_experiment()
