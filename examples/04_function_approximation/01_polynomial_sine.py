"""
Example 07-01 - Polynomial Approximation of a Target Function (sin(x))

This example demonstrates the use of evolutionary optimization to approximate a
mathematical target function using polynomial regression. Each individual represents the
coefficients of a polynomial. The objective is to minimize the mean squared error
between the target and the approximated function.

This example uses a plain monomial polynomial representation on [0, 2π].
Such polynomials are known to be numerically unstable, especially at higher
degrees. The instability is intentional: it illustrates how representation
choices affect the success of evolutionary optimization.

Experiment with different polynomial degrees or scaling the input domain to
[-1, 1] to observe the differences.

Note:
    Reproducibility is controlled via the `random_seed` field in the YAML config.
    Set it to an integer for deterministic runs or to null/omit it for stochastic runs.
"""

import numpy as np

from evolib import Indiv, Pop, plot_approximation

# Configuration
TARGET_FUNC = np.sin
x_cheb = np.cos(np.linspace(np.pi, 0, 200))
X_RANGE = (x_cheb + 1) * np.pi  # transform to [0, 2π]
FRAME_FOLDER = "01_frames_poly"
CONFIG_FILE = "01_polynomial_sine.yaml"


# Fitness Function
def fitness_function(indiv: Indiv) -> None:
    predicted = np.polyval(
        indiv.para["poly"].vector[::-1], X_RANGE
    )  # numpy expects highest degree first
    true_vals = TARGET_FUNC(X_RANGE)
    weights = 1.0 + 0.4 * np.abs(np.cos(X_RANGE))
    indiv.fitness = np.average((true_vals - predicted) ** 2, weights=weights)


def save_plot(pop: Pop) -> None:
    y_pred = np.polyval(pop.best().para["poly"].vector[::-1], X_RANGE)
    y_true = TARGET_FUNC(X_RANGE)
    plot_approximation(
        y_pred,
        y_true,
        title=f"Function approximation (gen={pop.generation_num}, "
        f"MSE={pop.best().fitness:.4f})",
        pred_marker=None,
        true_marker=None,
        pred_label="Approximation",
        show=False,
        show_grid=False,
        save_path=f"{FRAME_FOLDER}/gen_{pop.generation_num:03d}.png",
        x_vals=X_RANGE,
        y_limits=(-1.2, 1.2),
    )


# Main
def run_experiment() -> None:
    pop = Pop(CONFIG_FILE, fitness_function=fitness_function)
    pop.run(verbosity=1, on_generation=save_plot)


if __name__ == "__main__":
    run_experiment()
