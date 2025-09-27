# SPDX-License-Identifier: MIT
"""Common mathematical benchmark functions for optimization tasks."""

from typing import Literal, Sequence

import numpy as np
from numpy.random import default_rng


def generate_timeseries(
    length: int,
    normalize: bool = True,
    pattern: Literal[
        "default", "trend_switch", "parabolic", "zigzag", "shock"
    ] = "default",
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate synthetic time series data for evolution or forecasting. Uses local RNG to
    avoid modifying global numpy random state.

    Args:
        length (int): Number of time steps.
        normalize (bool): Whether to scale the output to [-1, 1].
        pattern (str): Pattern to generate:
            - "default": Trend + Sinus + Noise
            - "trend_switch": Linear trend up, then down
            - "parabolic": Smooth U-shaped curve (trend reversal)
            - "zigzag": Periodic linear up/down pattern
            - "shock": Trend followed by sharp reversal

    Returns:
        np.ndarray: The generated time series.
    """

    if seed is not None:
        rng = default_rng(seed)
    else:
        # Use global np.random set via config
        rng = np.random.default_rng(np.random.randint(0, 2**32 - 1))

    t = np.arange(length)

    if pattern == "shock":
        # Random switch point + optional shock slope variation
        switch_point = rng.integers(length // 3, 2 * length // 3)
        slope_up = rng.uniform(0.005, 0.015)
        slope_down = rng.uniform(-0.015, -0.005)
        shock = np.where(
            t < switch_point, slope_up * t, slope_down * (t - switch_point)
        )
        phase = rng.uniform(0, 2 * np.pi)
        seasonal = np.sin(t * 0.1 + phase)
        noise = rng.normal(0.05, 0.1, size=length)
        series = shock + seasonal + noise

    elif pattern == "parabolic":
        # Random shift of parabola + curvature
        center = rng.integers(length // 3, 2 * length // 3)
        curvature = rng.uniform(0.0002, 0.0004)
        trend = -curvature * (t - center) ** 2 + 1.0
        phase = rng.uniform(0, 2 * np.pi)
        seasonal = 0.5 * np.sin(t * 0.1 + phase)
        noise = rng.normal(0, 0.02, size=length)
        series = trend + seasonal + noise

    elif pattern == "zigzag":
        # Random zigzag period and slope
        period = rng.integers(20, 60)
        slope = rng.uniform(0.008, 0.015)
        trend = slope * ((t // period) % 2 * 2 - 1) * (t % period)
        phase = rng.uniform(0, 2 * np.pi)
        seasonal = 0.3 * np.sin(t * 0.2 + phase)
        noise = rng.normal(0, 0.03, size=length)
        series = trend + seasonal + noise

    elif pattern == "trend_switch":
        # Random trend_switch position + strength
        trend_switch_pos = rng.integers(length // 3, 2 * length // 3)
        trend_switch_strength = rng.uniform(-0.04, -0.02)
        trend = 0.01 * t
        trend_switch = np.where(
            t > trend_switch_pos, trend_switch_strength * (t - trend_switch_pos), 0.0
        )
        phase = rng.uniform(0, 2 * np.pi)
        seasonal = 0.5 * np.sin(t * 0.1 + phase)
        noise = rng.normal(0, 0.04, size=length)
        series = trend + trend_switch + seasonal + noise

    else:  # "default"
        # Minor phase + trend variation
        slope = rng.uniform(0.008, 0.012)
        phase = rng.uniform(0, 2 * np.pi)
        trend = slope * t
        seasonal = np.sin(t * 0.1 + phase)
        noise = rng.normal(0, 0.05, size=length)
        series = trend + seasonal + noise

    if normalize:  # Normalize to [-1, 1]
        min_val = np.min(series)
        max_val = np.max(series)
        series = 2 * (series - min_val) / (max_val - min_val) - 1

    return series


def lfsr_sequence(
    length: int,
    seed: int = 0b10011,
    taps: tuple[int, ...] = (5, 2),
    invert_feedback: bool = True,
) -> list[int]:
    """
    Generate a binary sequence using an n-bit LFSR (Linear Feedback Shift Register).

    Parameters
    ----------
    length : int
        Number of output bits to generate.
    seed : int, optional
        Initial state of the LFSR (must be non-zero). Default: 0b10011.
    taps : tuple[int, ...], optional
        Tap positions (1-based, e.g. (5, 2) for x^5 + x^2 + 1).
    invert_feedback : bool, optional
        If True (default), the feedback bit is inverted at initialization.
        This variant ensures a maximal-length sequence for (5, 2).
        If False, the classical Fibonacci-LFSR update rule is used.

    Returns
    -------
    list[int]
        Generated bit sequence of the given length.
    """
    n = max(taps)
    if seed == 0:
        raise ValueError("Seed must be non-zero for LFSR.")

    state = seed & ((1 << n) - 1)
    seq: list[int] = []

    for _ in range(length):
        bit = state & 1
        seq.append(bit)

        fb = 1 if invert_feedback else 0
        for t in taps:
            fb ^= (state >> (t - 1)) & 1

        state = (state >> 1) | (fb << (n - 1))

    return seq


def xor_sequence(length: int, k: int = 2, seed: Sequence[int] = (0, 1)) -> list[int]:
    """
    Generate a sequence where each new bit is the XOR of the last k bits.

    Parameters
    ----------
    length : int
        Length of the sequence to generate.
    k : int, optional
        Window size for XOR. Default: 2 (classic XOR sequence).
    seed : Sequence[int], optional
        Initial bits to start the sequence. Must have length >= k.

    Returns
    -------
    list[int]
        Generated bit sequence of the given length.
    """
    if len(seed) < k:
        raise ValueError(f"Seed must have at least {k} bits.")
    seq = list(seed)
    while len(seq) < length:
        next_bit = 0
        for i in range(1, k + 1):
            next_bit ^= seq[-i]
        seq.append(next_bit)
    return seq[:length]


def random_fixed_sequence(length: int, seed: int = 1234) -> list[int]:
    """
    Generate a reproducible random bit sequence with a fixed seed.

    Parameters
    ----------
    length : int
        Length of the sequence.
    seed : int, optional
        Random seed for reproducibility. Default: 1234.

    Returns
    -------
    list[int]
        Generated random bit sequence (0/1).
    """
    rng = np.random.RandomState(seed)
    return rng.randint(0, 2, size=length).tolist()


def simple_quadratic(x: np.ndarray) -> float:
    """
    Simple 1D benchmark: f(x) = x^2

    Global minimum: f(0) = 0

    Args:
        x (float or np.ndarray): Input value(s).

    Returns:
        float: Function value.
    """
    x = np.asarray(x, dtype=np.float64)
    return np.sum(x**2)


def rastrigin(x: np.ndarray, A: int = 10) -> float:
    """
    Rastrigin-Funktion (n-dimensional).

    Globales Minimum: f(0, ..., 0) = 0
    Empfohlener Suchraum: x_i ∈ [-5.12, 5.12]

    Args:
        x (list or np.ndarray): Eingabevektor (beliebige Dimension).
        A (float): Konstante der Rastrigin-Funktion (Standard: 10).

    Returns:
        float: Funktionswert der Rastrigin-Funktion an der Stelle x.
    """
    x = np.asarray(x, dtype=np.float64)

    if x.ndim == 0:  # Skalar → Vektor mit 1 Element
        x = x.reshape(1)
    elif x.ndim > 1:
        raise ValueError("Input x must be 1D or scalar.")

    n = x.size
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))


def sphere(x: np.ndarray) -> float:
    """
    Sphere function (n-dimensional).

    Global minimum: f(0, ..., 0) = 0
    Recommended domain: x_i ∈ [-5.12, 5.12]

    Args:
        x (list or np.ndarray): Input vector.

    Returns:
        float: Function value at x.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0:
        x = x.reshape(1)
    elif x.ndim > 1:
        raise ValueError("Input x must be 1D or scalar.")

    return np.sum(x**2)


def rosenbrock(x: np.ndarray) -> float:
    """
    Rosenbrock function (n-dimensional).

    Global minimum: f(1, ..., 1) = 0
    Recommended domain: x_i ∈ [-5, 10]

    Args:
        x (list or np.ndarray): Input vector.

    Returns:
        float: Function value at x.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0:
        x = x.reshape(1)
    elif x.ndim > 1:
        raise ValueError("Input x must be 1D or scalar.")
    if len(x) < 2:
        raise ValueError("Rosenbrock needs at least 2 dimensions")

    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def ackley(x: np.ndarray) -> float:
    """
    Ackley function (n-dimensional).

    Global minimum: f(0, ..., 0) = 0
    Recommended domain: x_i ∈ [-32.768, 32.768]

    Args:
        x (list or np.ndarray): Input vector.

    Returns:
        float: Function value at x.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0:
        x = x.reshape(1)
    elif x.ndim > 1:
        raise ValueError("Input x must be 1D or scalar.")

    n = x.size
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + 20 + np.e


def griewank(x: np.ndarray) -> float:
    """
    Griewank function (n-dimensional).

    Global minimum: f(0, ..., 0) = 0
    Recommended domain: x_i ∈ [-600, 600]

    Args:
        x (list or np.ndarray): Input vector.

    Returns:
        float: Function value at x.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0:
        x = x.reshape(1)
    elif x.ndim > 1:
        raise ValueError("Input x must be 1D or scalar.")

    sum_sq = np.sum(x**2) / 4000
    prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, x.size + 1))))
    return sum_sq - prod_cos + 1


def schwefel(x: np.ndarray) -> float:
    """
    Schwefel function (n-dimensional).

    Global minimum: f(420.9687, ..., 420.9687) = 0
    Recommended domain: x_i ∈ [-500, 500]

    Args:
        x (list or np.ndarray): Input vector.

    Returns:
        float: Function value at x.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0:
        x = x.reshape(1)
    elif x.ndim > 1:
        raise ValueError("Input x must be 1D or scalar.")

    return 418.9829 * x.size - np.sum(x * np.sin(np.sqrt(np.abs(x))))


def rosenbrock_2d(
    x: float | np.ndarray,
    y: float | np.ndarray,
) -> float | np.ndarray:
    """
    2D Rosenbrock function.

    Global minimum at (1, 1) with f(x, y) = 0.

    Args:
        x (float or np.ndarray): x-coordinate(s)
        y (float or np.ndarray): y-coordinate(s)

    Returns:
        float or np.ndarray: function value(s)
    """
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


def rastrigin_2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    2D Rastrigin function, evaluated element-wise on meshgrid arrays.

    Global minimum at (0, 0) with f(x, y) = 0.
    Highly multimodal.

    Args:
        x (np.ndarray): Meshgrid-style array of x-values.
        y (np.ndarray): Meshgrid-style array of y-values.

    Returns:
        np.ndarray: Fitness values for each (x, y) pair.
    """
    A = 10
    return (
        A * 2 + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))
    )


def griewank_2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    2D Griewank function, evaluated element-wise on meshgrid arrays.

    Global minimum at (0, 0) with f(x, y) = 0.
    Non-convex, with moderate multimodality.

    Args:
        x (np.ndarray): Meshgrid-style array of x-values.
        y (np.ndarray): Meshgrid-style array of y-values.

    Returns:
        np.ndarray: Fitness values for each (x, y) pair.
    """
    part1 = (x**2 + y**2) / 4000
    part2 = np.cos(x) * np.cos(y / np.sqrt(2))
    return part1 - part2 + 1


def sphere_2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    2D Sphere function, evaluated element-wise on meshgrid arrays.

    Global minimum at (0, 0) with f(x, y) = 0.
    Convex and unimodal.

    Args:
        x (np.ndarray): Meshgrid-style array of x-values.
        y (np.ndarray): Meshgrid-style array of y-values.

    Returns:
        np.ndarray: Fitness values for each (x, y) pair.
    """
    return x**2 + y**2


def schwefel_2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    2D Schwefel function, evaluated element-wise on meshgrid arrays.

    Global minimum at (420.9687, 420.9687) with f(x, y) = 0.
    Many local minima; highly deceptive.

    Args:
        x (np.ndarray): Meshgrid-style array of x-values.
        y (np.ndarray): Meshgrid-style array of y-values.

    Returns:
        np.ndarray: Fitness values for each (x, y) pair.
    """
    return 418.9829 * 2 - (
        x * np.sin(np.sqrt(np.abs(x))) + y * np.sin(np.sqrt(np.abs(y)))
    )


def ackley_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    3D Ackley function evaluated on meshgrid-style inputs.

    Global minimum at (0, 0, 0).
    """
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = 3
    sum_sq = x**2 + y**2 + z**2
    sum_cos = np.cos(c * x) + np.cos(c * y) + np.cos(c * z)
    term1 = -a * np.exp(-b * np.sqrt(sum_sq / d))
    term2 = -np.exp(sum_cos / d)
    return term1 + term2 + a + np.exp(1)


def rastrigin_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    3D Rastrigin function.

    Global minimum at (0, 0, 0).
    """
    A = 10
    return (
        A * 3
        + (x**2 - A * np.cos(2 * np.pi * x))
        + (y**2 - A * np.cos(2 * np.pi * y))
        + (z**2 - A * np.cos(2 * np.pi * z))
    )


def griewank_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    3D Griewank function.

    Global minimum at (0, 0, 0).
    """
    part1 = (x**2 + y**2 + z**2) / 4000
    part2 = np.cos(x) * np.cos(y / np.sqrt(2)) * np.cos(z / np.sqrt(3))
    return part1 - part2 + 1


def sphere_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    3D Sphere function.

    Global minimum at (0, 0, 0).
    """
    return x**2 + y**2 + z**2


def rosenbrock_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    3D Rosenbrock function.

    Minimum at (1, 1, 1).
    """
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2 + (1 - y) ** 2 + 100 * (z - y**2) ** 2


def schwefel_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    3D Schwefel function.

    Minimum at (420.9687, 420.9687, 420.9687).
    """
    return 418.9829 * 3 - (
        x * np.sin(np.sqrt(np.abs(x)))
        + y * np.sin(np.sqrt(np.abs(y)))
        + z * np.sin(np.sqrt(np.abs(z)))
    )


def ackley_2d(
    x: float | np.ndarray,
    y: float | np.ndarray,
    a: float = 20,
    b: float = 0.2,
    c: float = 2 * np.pi,
) -> float | np.ndarray:
    """
    2D Ackley test function.

    Global minimum at (0, 0): f(0,0) = 0

    Args:
        x (float | np.ndarray): x-coordinate(s)
        y (float | np.ndarray): y-coordinate(s)
        a, b, c (float): Ackley function parameters

    Returns:
        float | np.ndarray: function value(s)
    """
    term1 = -a * np.exp(-b * np.sqrt(0.5 * (x**2 + y**2)))
    term2 = -np.exp(0.5 * (np.cos(c * x) + np.cos(c * y)))
    return term1 + term2 + a + np.exp(1)
