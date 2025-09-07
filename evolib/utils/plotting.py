# SPDX-License-Identifier: MIT
"""
Plotting utilities for visualizing evolutionary progress.

This module provides functions for visualizing:
- Fitness statistics (best, mean, median, std)
- Diversity over time
- Mutation probability and strength trends
- Fitness comparison
"""
from __future__ import annotations

import os
import tempfile
import warnings
from typing import Literal, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from evonet.core import Nnet
from PIL import Image

from evolib.core.population import Pop

PopOrDf = Union["Pop", pd.DataFrame]


def _as_history_df(obj: PopOrDf) -> pd.DataFrame:
    return obj.history_df if hasattr(obj, "history_df") else obj


def plot_history(
    histories: Sequence[PopOrDf] | Pop,
    *,
    metrics: list[str] = ["best_fitness"],
    labels: Optional[list[str]] = None,
    title: str = "Evolutionary Metrics",
    xlabel: str = "Generation",
    ylabel: Optional[str] = None,
    show: bool = True,
    log_y: bool = False,
    save_path: Optional[str] = None,
    with_std: bool = True,
    figsize: tuple = (10, 6),
) -> None:
    """
    General-purpose plotting function for evolutionary history metrics.

    Supports multiple metrics (e.g. fitness, diversity, mutation )probability and
    comparison across runs.

    Args:
        histories (pd.DataFrame | list[pd.DataFrame]): Single or multiple history
            DataFrames.
        metrics (list[str]): List of metric column names to plot (e.g., 'best_fitness').
        labels (list[str], optional): Optional list of labels for each run.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str | None): Label for the y-axis (auto-generated if None).
        show (bool): Whether to display the plot interactively.
        log_y (bool): Apply logarithmic scale to the y-axis.
        save_path (str | None): Optional file path to save the plot.
        with_std (bool): If True, plot standard deviation shading when available.
        figsize (tuple): Size of the figure (width, height).
    """

    # Normalize input to list
    if isinstance(histories, Pop):
        histories = [histories]
    elif isinstance(histories, pd.DataFrame):
        histories = [histories]

    dfs = [_as_history_df(h) for h in histories]

    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(dfs))]

    plt.figure(figsize=figsize)

    if log_y:
        plt.yscale("log")

    for hist, label in zip(dfs, labels):
        generations = hist["generation"]

        for metric in metrics:
            if metric not in hist.columns:
                warnings.warn(
                    f"Metric '{metric}' not found in history for '{label}' — skipping."
                )
                continue

            line_label = f"{label} - {metric}" if len(dfs) > 1 else metric
            plt.plot(generations, hist[metric], label=line_label)

            # Optional standard deviation band if available
            if with_std and "mean" in metric:
                std_col = metric.replace("mean", "std")
                if std_col in hist.columns:
                    lower = hist[metric] - hist[std_col]
                    upper = hist[metric] + hist[std_col]
                    plt.fill_between(generations, lower, upper, alpha=0.2)

    plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    elif len(metrics) == 1:
        plt.ylabel(metrics[0].replace("_", " ").capitalize())
    else:
        plt.ylabel("Metric Value")

    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to '{save_path}'")

    if show:
        plt.show()
    else:
        plt.close()


def plot_fitness(
    history: PopOrDf,
    *,
    title: str = "Fitness over Generations",
    show: bool = True,
    log: bool = False,
    save_path: Optional[str] = None,
) -> None:
    """Wrapper to plot best, mean, and median fitness with optional std band."""

    df = _as_history_df(history)

    plot_history(
        df,
        metrics=["best_fitness", "mean_fitness", "median_fitness"],
        title=title,
        log_y=log,
        ylabel="Fitness",
        save_path=save_path,
        show=show,
        with_std=True,
    )


def plot_diversity(
    history: PopOrDf,
    *,
    title: str = "Population Diversity",
    show: bool = True,
    log: bool = False,
    save_path: Optional[str] = None,
) -> None:
    """Wrapper to plot diversity over generations."""

    df = _as_history_df(history)

    if "diversity" not in df.columns:
        warnings.warn("Column 'diversity' not found in history.")
        return

    plot_history(
        df,
        metrics=["diversity"],
        title=title,
        log_y=log,
        ylabel="Diversity",
        save_path=save_path,
        show=show,
        with_std=False,
    )


def plot_mutation_trends(
    history: PopOrDf,
    *,
    title: str = "Mutation Parameter Trends",
    show: bool = True,
    log: bool = False,
    save_path: Optional[str] = None,
) -> None:
    """Wrapper to plot mutation and/or strength trends over time."""

    df = _as_history_df(history)

    metrics = []
    if "mutation_probability_mean" in df.columns:
        metrics.append("mutation_probability_mean")
    if "mutation_strength_mean" in df.columns:
        metrics.append("mutation_strength_mean")

    if not metrics:
        warnings.warn("No mutation-related columns found in history.")
        return

    plot_history(
        df,
        metrics=metrics,
        title=title,
        log_y=log,
        ylabel="Mutation Parameters",
        save_path=save_path,
        show=show,
        with_std=False,
    )


def plot_fitness_comparison(
    histories: Sequence[PopOrDf],
    *,
    labels: Optional[list[str]] = None,
    metric: str = "best_fitness",
    title: str = "Fitness Comparison over Generations",
    show: bool = True,
    log: bool = False,
    save_path: Optional[str] = None,
) -> None:

    if not isinstance(histories, (list, tuple)):
        raise TypeError("histories must be a List or a Tuple")

    dfs = [_as_history_df(h) for h in histories]

    # Skip histories that lack the requested metric
    filtered = []
    filtered_labels = []
    for i, hist in enumerate(dfs):
        if metric in hist.columns:
            filtered.append(hist)
            filtered_labels.append(labels[i] if labels else f"Run {i+1}")
        else:
            warnings.warn(f"Metric '{metric}' not found in run {i+1}. Skipping.")

    if not filtered:
        warnings.warn("No valid runs to plot.")
        return

    plot_history(
        filtered,
        metrics=[metric],
        labels=filtered_labels,
        title=title,
        log_y=log,
        save_path=save_path,
        show=show,
        with_std=True,
    )


def plot_approximation(
    y_pred: Sequence[float] | np.ndarray,
    y_true: Sequence[float] | np.ndarray,
    *,
    title: str = "Approximation",
    show: bool = True,
    show_grid: bool = True,
    legend_location: str = "upper right",
    save_path: Optional[str] = None,
    pred_label: str = "Prediction",
    true_label: str = "Target",
    pred_marker: str | None = None,
    true_marker: str | None = None,
    x_vals: Sequence[float] | np.ndarray | None = None,
    y_limits: tuple[float, float] | None = None,
    dpi: int = 150,
    size: tuple[float, float] = (6, 4),
    fitness: float | None = None,
    support_points: (
        tuple[Sequence[float] | np.ndarray, Sequence[float] | np.ndarray] | None
    ) = None,
    support_label: str = "Support points",
    support_marker: str = "o",
    support_size: float = 30.0,
    xlabel: str | None = None,
    ylabel: str | None = None,
    residuals_style: Optional[Literal["overlay", "subplot"]] = None,
    residuals_label: str = "Residuals (target - prediction)",
    residuals_alpha: float = 0.5,
    residuals_linewidth: float = 1.0,
) -> None:
    """
    Plot predicted values against targets with optional support points and residual
    visualization.

    Args:
        y_pred: Predicted values (1D).
        y_true: Target values (1D), same length as ``y_pred``.
        title: Plot title. If ``fitness`` is given, it is appended.
        show: If True, show the plot interactively.
        show_grid: Toggle background grid.
        legend_location: Matplotlib legend location string.
        save_path: If provided, save the figure to this path.
        pred_label: Legend label for predictions.
        true_label: Legend label for targets.
        x_vals: Optional x-axis values; defaults to ``range(len(y_true))``.
        y_limits: Optional (ymin, ymax) for the main plot; if None, computed
                  with padding.
        dpi: Figure DPI.
        size: Figure size (width, height) in inches.
        fitness: Optional fitness to append to the title (e.g. MSE).

        support_points: Optional (x_support, y_support) to visualize discrete
                        support points.
        support_label: Legend label for support points.
        support_marker: Marker symbol for support points.
        support_size: Marker size for support points.
        xlabel, ylabel: Axis labels for the main plot.

        residuals_style: "subplot" for a second axes below, or "overlay" to draw error
                        segments in the main axes.
        residuals_label: Legend label for residuals (subplot mode).
        residuals_alpha: Transparency for residual drawing.
        residuals_linewidth: Line width for residual drawing.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} " f"vs y_pred {y_pred.shape}"
        )

    if x_vals is None:
        x_vals = np.arange(len(y_true), dtype=float)
    else:
        x_vals = np.asarray(x_vals, dtype=float).ravel()
        if x_vals.shape[0] != y_true.shape[0]:
            raise ValueError(
                f"Length mismatch: x_vals {x_vals.shape[0]} " f"vs y {y_true.shape[0]}"
            )

    if fitness is not None:
        title = f"{title} (fitness={fitness:.4f})"

    # figure/axes creation
    if residuals_style == "subplot":
        fig, (ax, ax_res) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(size[0], size[1] * 1.4),
            dpi=dpi,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
            constrained_layout=True,
        )
        use_tight = False
    else:
        fig, ax = plt.subplots(figsize=size, dpi=dpi)
        ax_res = None
        use_tight = True

    # main plot
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.plot(x_vals, y_true, color="black", label=true_label, marker=true_marker, lw=2)
    ax.plot(
        x_vals, y_pred, color="red", label=pred_label, marker=pred_marker, lw=2, ls="--"
    )

    if support_points is not None:
        sx, sy = support_points
        sx = np.asarray(sx, dtype=float).ravel()
        sy = np.asarray(sy, dtype=float).ravel()
        if sx.shape != sy.shape:
            raise ValueError(
                f"Support points shape mismatch: x {sx.shape} " f"vs y {sy.shape}"
            )
        ax.scatter(
            sx, sy, label=support_label, s=support_size, marker=support_marker, zorder=3
        )

    ax.legend(loc=legend_location)
    ax.grid(show_grid)

    # y-limits (main)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    else:
        y_all = [y_true, y_pred]
        if support_points is not None:
            y_all.append(np.asarray(support_points[1], dtype=float).ravel())
        y_all = np.concatenate(y_all)
        pad = 0.05 * (y_all.max() - y_all.min() + 1e-12)
        ax.set_ylim(y_all.min() - pad, y_all.max() + pad)

    # residuals
    if residuals_style is not None:
        residuals = y_true - y_pred
        if residuals_style == "subplot":
            ax_res.plot(
                x_vals, residuals, lw=1.5, alpha=residuals_alpha, label=residuals_label
            )
            ax_res.axhline(0.0, lw=1.0, alpha=0.7)
            ax_res.set_ylabel("resid")
            ax_res.grid(show_grid)
            ax_res.legend(loc="upper right")
        elif residuals_style == "overlay":
            y_min = np.minimum(y_true, y_pred)
            y_max = np.maximum(y_true, y_pred)
            ax.vlines(
                x_vals,
                y_min,
                y_max,
                alpha=residuals_alpha,
                linewidth=residuals_linewidth,
            )
        else:
            raise ValueError('residuals_style must be "subplot" or "overlay"')

    # layout handling
    if use_tight:
        fig.tight_layout()

    # saving
    if save_path:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        fig.savefig(save_path, dpi=dpi)

    # show/close
    if show:
        plt.show()
    else:
        plt.close(fig)


def save_combined_net_plot(
    net: Nnet,
    X: np.ndarray,
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    path: str,
    *,
    dpi: int = 100,
    title: str = "EvoNet Fit to sin(x)",
) -> None:
    """
    Saves a combined image of network structure and approximation plot.

    Args:
        net: EvoNet instance with .print_graph().
        X (np.ndarray): Input values (for plotting).
        Y_true (np.ndarray): Ground truth values (e.g. sin(x)).
        Y_pred (np.ndarray): Network prediction values.
        path (str): Output path for PNG image.
        dpi (int): DPI resolution for the output.
        title (str): Plot title.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        graph_path = os.path.join(tmpdir, "net")
        net.print_graph(graph_path, fillcolors_on=True, thickness_on=True)
        img_net = Image.open(graph_path + ".png")

        # Create approximation plot
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(X, Y_true, label="Target: sin(x)")
        ax.plot(X, Y_pred, label="Network Output", linestyle="--")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        ax.grid(True)
        ax.legend()
        fig.tight_layout()

        plot_path = os.path.join(tmpdir, "plot.png")
        fig.savefig(plot_path, dpi=dpi)
        plt.close(fig)

        img_plot = Image.open(plot_path)

        # Combine horizontally
        total_width = img_net.width + img_plot.width
        height = max(img_net.height, img_plot.height)

        combined = Image.new("RGB", (total_width, height), "white")
        combined.paste(img_net, (0, 0))
        combined.paste(img_plot, (img_net.width, 0))
        combined.save(path)


def save_current_plot(filename: str, dpi: int = 300) -> None:
    """Save the current matplotlib figure to file."""
    plt.savefig(filename, dpi=dpi)
    print(f"Plot saved to '{filename}'")
