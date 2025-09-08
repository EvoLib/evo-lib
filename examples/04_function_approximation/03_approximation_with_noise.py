"""
Example 04-03 – Approximation with Noisy Data.

This example shows how an evolutionary approximation behaves when the target signal is
corrupted by Gaussian noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from evolib import FitnessFunction, Indiv, Pop, plot_approximation

# Evaluation grid (dense)
X_EVAL = np.linspace(0, 2 * np.pi, 400)

# Noise
np.random.seed(42)
NOISE_STD = 0.10

FRAME_FOLDER = "03_frames_noise"
CONFIG_FILE = "03_approximation_with_noise.yaml"


@dataclass
class NoiseContext:
    """Holds the current noisy target sample for evaluation/plotting."""

    current_noise: np.ndarray


def noisy_target(x: np.ndarray) -> np.ndarray:
    """Return sin(x) plus Gaussian noise (uses global RNG seed for reproducibility)."""
    return np.sin(x) + np.random.normal(0.0, NOISE_STD, size=x.size)


def make_set_noise(ctx: NoiseContext) -> Callable[[Pop], None]:
    """Factory for the `on_generation_start` callback that refreshes noisy target."""

    def set_noise(_: Pop) -> None:
        ctx.current_noise = noisy_target(X_EVAL)

    return set_noise


class NoisyMSEFitness(FitnessFunction):
    """MSE against the current noisy target."""

    def __init__(self, ctx: NoiseContext) -> None:
        self.ctx = ctx

    def __call__(self, indiv: Indiv) -> None:
        # Derive support grid from this individual's dimension
        dim = indiv.para["points"].dim
        x_support = np.linspace(0, 2 * np.pi, dim)

        y_support = indiv.para["points"].vector
        y_pred = np.interp(X_EVAL, x_support, y_support)

        mse = float(np.mean((self.ctx.current_noise - y_pred) ** 2))
        indiv.fitness = mse


def make_save_plot(ctx: NoiseContext) -> Callable[[Pop], None]:
    """Factory for the `on_generation_end` callback that saves a comparison plot."""

    def save_plot(pop: Pop) -> None:
        best = pop.best()
        dim = best.para["points"].dim
        x_support = np.linspace(0, 2 * np.pi, dim)
        y_support = best.para["points"].vector
        y_pred = np.interp(X_EVAL, x_support, y_support)
        y_true = np.sin(X_EVAL)

        title = (
            f"Noisy fit (gen={pop.generation_num}, "
            f"MSE={best.fitness:.4f}, sigma={NOISE_STD})"
        )
        plot_approximation(
            y_pred=y_pred,
            y_true=y_true,
            title=title,
            x_vals=X_EVAL,
            y_limits=(-1.6, 1.6),
            pred_label="Approximation",
            show=False,
            show_grid=False,
            save_path=f"{FRAME_FOLDER}/gen_{pop.generation_num:03d}.png",
            support_points=(x_support, y_support),
            residuals_style="subplot",  # or "overlay"
            residuals_ylimits=(-1.0, 1.0),
            extra_lines=[
                (X_EVAL, ctx.current_noise, "Noisy target", {"ls": ":", "alpha": 0.6})
            ],
        )

    return save_plot


def run_experiment() -> None:
    # Initialize context with one noisy sample so that it's available for gen=0 plotting
    ctx = NoiseContext(current_noise=noisy_target(X_EVAL))

    pop = Pop(CONFIG_FILE)
    pop.set_fitness_function(NoisyMSEFitness(ctx))
    pop.run(
        verbosity=1,
        on_generation_start=make_set_noise(ctx),
        on_generation_end=make_save_plot(ctx),
    )


if __name__ == "__main__":
    run_experiment()
