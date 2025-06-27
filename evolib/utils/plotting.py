# SPDX-License-Identifier: MIT
"""
Plotting utilities for visualizing evolutionary progress.

This module provides functions for visualizing:
- Fitness statistics (best, mean, median, std)
- Diversity over time
- Mutation probability and strength trends
- Fitness comparison
"""

from typing import Optional, Union

import matplotlib.pyplot as plt
import pandas as pd


def plot_history(
    histories: Union[pd.DataFrame, list[pd.DataFrame]],
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
    if isinstance(histories, pd.DataFrame):
        histories = [histories]

    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(histories))]

    plt.figure(figsize=figsize)

    if log_y:
        plt.yscale("log")

    for hist, label in zip(histories, labels):
        generations = hist["generation"]

        for metric in metrics:
            if metric not in hist.columns:
                print(
                    f"Metric '{metric}' not found in history for '{label}' — skipping."
                )
                continue

            line_label = f"{label} - {metric}" if len(histories) > 1 else metric
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
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to '{save_path}'")

    if show:
        plt.show()
    else:
        plt.close()


def plot_fitness(
    history: pd.DataFrame,
    *,
    title: str = "Fitness over Generations",
    show: bool = True,
    log: bool = False,
    save_path: Optional[str] = None,
) -> None:
    """Wrapper to plot best, mean, and median fitness with optional std band."""
    plot_history(
        history,
        metrics=["best_fitness", "mean_fitness", "median_fitness"],
        title=title,
        log_y=log,
        save_path=save_path,
        show=show,
        with_std=True,
    )


def plot_diversity(
    history: pd.DataFrame,
    *,
    title: str = "Population Diversity",
    show: bool = True,
    log: bool = False,
    save_path: Optional[str] = None,
) -> None:
    """Wrapper to plot diversity over generations."""
    if "diversity" not in history.columns:
        print("Column 'diversity' not found in history.")
        return

    plot_history(
        history,
        metrics=["diversity"],
        title=title,
        log_y=log,
        save_path=save_path,
        show=show,
        with_std=False,
    )


def plot_mutation_trends(
    history: pd.DataFrame,
    *,
    title: str = "Mutation Parameter Trends",
    show: bool = True,
    log: bool = False,
    save_path: Optional[str] = None,
) -> None:
    """Wrapper to plot mutation and/or strength trends over time."""
    metrics = []
    if "mutation_probability_mean" in history.columns:
        metrics.append("mutation_probability_mean")
    if "mutation_strength_mean" in history.columns:
        metrics.append("mutation_strength_mean")

    if not metrics:
        print("No mutation-related columns found in history.")
        return

    plot_history(
        history,
        metrics=metrics,
        title=title,
        log_y=log,
        save_path=save_path,
        show=show,
        with_std=False,
    )


def plot_fitness_comparison(
    histories: list[pd.DataFrame],
    *,
    labels: Optional[list[str]] = None,
    metric: str = "best_fitness",
    title: str = "Fitness Comparison over Generations",
    show: bool = True,
    log: bool = False,
    save_path: Optional[str] = None,
) -> None:
    """Wrapper to compare a fitness metric across multiple runs."""
    # Skip histories that lack the requested metric
    filtered = []
    filtered_labels = []
    for i, hist in enumerate(histories):
        if metric in hist.columns:
            filtered.append(hist)
            filtered_labels.append(labels[i] if labels else f"Run {i+1}")
        else:
            print(f"Metric '{metric}' not found in run {i+1}. Skipping.")

    if not filtered:
        print("No valid runs to plot.")
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


def save_current_plot(filename: str, dpi: int = 300) -> None:
    """Save the current matplotlib figure to file."""
    plt.savefig(filename, dpi=dpi)
    print(f"Plot saved to '{filename}'")
