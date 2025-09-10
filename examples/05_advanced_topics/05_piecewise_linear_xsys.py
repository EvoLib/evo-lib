"""
Example 05-06 – Piecewise-linear approximation with learnable knots (xs + ys)

Two modules:
- 'xs': knot positions in [0, 2π] (sorted per individual)
- 'ys': knot values in [-1.5, 1.5]

Prediction: linear interpolation over (xs, ys) on a fine grid.
Fitness: MSE against a target function (e.g. sin or step).
Support points are plotted for interpretability.

Didactic focus:
- Two separate modules (geometry vs. values)
- Coordinate-based representation learning
- Support points = true interpolation "kinks"
"""

from __future__ import annotations

import os

import numpy as np

from evolib import Indiv, Pop, mse_loss, plot_approximation

# Evaluation grid and target function
X_FINE = np.linspace(0, 2 * np.pi, 200)

# --- Change target function here ---
# Y_TRUE = np.sin(X_FINE)  # smooth
# Y_TRUE = 2 * (X_FINE / (2 * np.pi)) - 1  # sawtooth
# Y_TRUE = np.exp(-4 * (X_FINE - 1.5)**2) + 0.6 * np.exp(-6 * (X_FINE - 4.5)**2) - 0.3
Y_TRUE = np.where(X_FINE < np.pi, 1.0, -1.0)  # step function (rectangle)
# -----------------------------------

FRAME_FOLDER = "05_frames_xs_ys"
CONFIG_FILE = "05_piecewise_linear_xsys.yaml"

SUPPORT_TOL = 1e-3  # Tolerance for filtering visually redundant support points


def piecewise_linear_predict(
    xs: np.ndarray, ys: np.ndarray, x_eval: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform linear interpolation based on (xs, ys) at evaluation points x_eval.

    Sorts xs internally and handles near-duplicates.
    """
    order = np.argsort(xs)
    xs_sorted = xs[order]
    ys_sorted = ys[order]

    # Slightly offset duplicate xs to avoid interpolation errors
    diff = np.diff(xs_sorted)
    if np.any(diff == 0.0):
        eps = 1e-6
        for i in range(1, len(xs_sorted)):
            if xs_sorted[i] <= xs_sorted[i - 1]:
                xs_sorted[i] = xs_sorted[i - 1] + eps

    y_pred = np.interp(
        x_eval, xs_sorted, ys_sorted, left=ys_sorted[0], right=ys_sorted[-1]
    )
    return y_pred, xs_sorted, ys_sorted


def fitness_function(indiv: Indiv) -> None:
    """Evaluate fitness as mean squared error between prediction and target function."""
    xs = np.array(indiv.para["xs"].vector, dtype=float)
    ys = np.array(indiv.para["ys"].vector, dtype=float)

    y_pred, _, _ = piecewise_linear_predict(xs, ys, X_FINE)
    mse = mse_loss(Y_TRUE, y_pred)

    indiv.fitness = mse


def save_plot(pop: Pop, filter_support_duplicates: bool = True) -> None:
    """
    Save approximation plot for the best individual of the current generation.

    Optionally filter out visually redundant support points.
    """
    if not os.path.exists(FRAME_FOLDER):
        os.makedirs(FRAME_FOLDER, exist_ok=True)

    best = pop.best()
    xs = np.asarray(best.para["xs"].vector, dtype=float)
    ys = np.asarray(best.para["ys"].vector, dtype=float)

    y_pred, xs_sorted, ys_sorted = piecewise_linear_predict(xs, ys, X_FINE)

    # Filter out near-duplicate support points (optional)
    if filter_support_duplicates:
        mask = np.append([True], np.diff(xs_sorted) > SUPPORT_TOL)
        x_support = xs_sorted[mask]
        y_support = ys_sorted[mask]
    else:
        x_support = xs_sorted
        y_support = ys_sorted

    title = (
        f"Piecewise-Linear (xs+ys) – gen={pop.generation_num}, "
        f"MSE={best.extra_metrics.get('mse', best.fitness):.4f}"
    )

    plot_approximation(
        y_pred=y_pred,
        y_true=Y_TRUE,
        title=title,
        x_vals=X_FINE,
        y_limits=(-1.6, 1.6),
        pred_label="Approximation",
        show=False,
        show_grid=False,
        save_path=f"{FRAME_FOLDER}/gen_{pop.generation_num:03d}.png",
        support_points=(x_support, y_support),
        residuals_style="subplot",
        residuals_ylimits=(-1.0, 1.0),
    )


def run_experiment() -> None:
    pop = Pop(CONFIG_FILE, fitness_function=fitness_function)
    pop.run(verbosity=1, on_generation_end=save_plot)


if __name__ == "__main__":
    run_experiment()
