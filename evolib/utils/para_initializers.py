# SPDX-License-Identifier: MIT
"""
para_initializers.py – Utility functions for generating initial parameter vectors.

This module provides callable factory functions to generate individual `para`
vectors for use in `Indiv(para=...)`, e.g., when initializing a population.

These initializers are especially useful for examples, tests, or general-purpose
evolution strategies working with real-valued vector representations.

Typical usage:
    pop.initialize_random_population(initializer=init_uniform_vector((-1, 1), dim=10))

Functions:
    init_uniform_vector(bounds, dim): Samples para from uniform distribution.
    init_normal_vector(mu, sigma, dim): Samples para from normal distribution.
    init_constant_vector(value, dim): Creates constant vector.
"""

from typing import Callable

import numpy as np


def init_uniform_vector(bounds: tuple[float, float], dim: int) -> Callable:
    """
    Return a function that samples a real-valued vector from a uniform distribution.

    Args:
        bounds (tuple[float, float]): Lower and upper bounds (inclusive).
        dim (int): Number of dimensions.

    Returns:
        Callable: A function pop -> para (list[float]) to be used in initialization.
    """

    def initializer(_: object) -> list[float]:
        return np.random.uniform(bounds[0], bounds[1], size=dim).tolist()

    return initializer


def init_normal_vector(mu: float, sigma: float, dim: int) -> Callable:
    """
    Return a function that samples a real-valued vector from a normal distribution.

    Args:
        mu (float): Mean of the normal distribution.
        sigma (float): Standard deviation.
        dim (int): Number of dimensions.

    Returns:
        Callable: A function pop -> para (list[float]) to be used in initialization.
    """

    def initializer(_: object) -> list[float]:
        return np.random.normal(loc=mu, scale=sigma, size=dim).tolist()

    return initializer


def init_constant_vector(value: float, dim: int) -> Callable:
    """
    Return a function that produces a constant vector.

    Args:
        value (float): Value for each entry in the vector.
        dim (int): Number of dimensions.

    Returns:
        Callable: A function pop -> para (list[float]) to be used in initialization.
    """

    def initializer(_: object) -> list[float]:
        return [value] * dim

    return initializer
