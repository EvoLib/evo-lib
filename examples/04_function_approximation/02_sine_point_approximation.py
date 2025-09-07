"""
Example 04-02 - Sine Approximation via supportpoints (Y-Vektoren)

Approximates sin(x) by optimizing Y-values at fixed X-support points using evolutionary
strategies. This approach avoids polynomial instability and works with any interpolation
method.
"""

import numpy as np

from evolib import Indiv, Pop, plot_approximation

# Parameters
X_DENSE = np.linspace(0, 2 * np.pi, 400)
Y_TRUE = np.sin(X_DENSE)

FRAME_FOLDER = "02_frames_point"
CONFIG_FILE = "02_sine_point_approximation.yaml"


# Fitness
def fitness_function(indiv: Indiv) -> None:
    # derive support grid from this individual's dimension
    dim = indiv.para["points"].dim
    x_support = np.linspace(0, 2 * np.pi, dim)

    y_support = indiv.para["points"].vector
    y_pred = np.interp(X_DENSE, x_support, y_support)

    # weighted MSE (didactic: emphasize near maxima/minima)
    weights = 1.0 + 0.4 * np.abs(np.cos(X_DENSE))
    indiv.fitness = np.average((Y_TRUE - y_pred) ** 2, weights=weights)


def save_plot(pop: Pop) -> None:
    best = pop.best()
    dim = best.para["points"].dim
    x_support = np.linspace(0, 2 * np.pi, dim)

    y_support = best.para["points"].vector
    y_pred = np.interp(X_DENSE, x_support, y_support)

    plot_approximation(
        y_pred,
        Y_TRUE,
        title=f"Function approximation (gen={pop.generation_num}, "
        f"MSE={best.fitness:.4f})",
        pred_label="Approximation",
        show=False,
        show_grid=False,
        save_path=f"{FRAME_FOLDER}/gen_{pop.generation_num:03d}.png",
        x_vals=X_DENSE,
        y_limits=(-1.2, 1.2),
        support_points=(x_support, y_support),
    )


# Main
def run_experiment() -> None:
    pop = Pop(CONFIG_FILE, fitness_function=fitness_function)
    pop.run(verbosity=1, on_generation=save_plot)


if __name__ == "__main__":
    run_experiment()
