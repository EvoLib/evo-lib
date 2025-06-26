# SPDX-License-Identifier: MIT
"""Common mathematical benchmark functions for optimization tasks."""

import numpy as np


def simple_quadratic(x: np.ndarray) -> np.ndarray:
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


def rastrigin(x: np.ndarray, A: int = 10) -> np.ndarray:
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


def sphere(x: np.ndarray) -> np.ndarray:
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


def rosenbrock(x: np.ndarray) -> np.ndarray:
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


def ackley(x: np.ndarray) -> np.ndarray:
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


def griewank(x: np.ndarray) -> np.ndarray:
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


def schwefel(x: np.ndarray) -> np.ndarray:
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
