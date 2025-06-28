# SPDX-License-Identifier: MIT
"""Initializers for ParaVector representations."""

from typing import Callable

import numpy as np
from core.population import Pop
from representation.para_vector import ParaVector


def random_initializer(
    dim: int, bounds: tuple[float, float] = (-1.0, 1.0), tau: float = 0.0
) -> Callable[[Pop], ParaVector]:
    """
    Returns a factory function that creates ParaVector instances with uniformly
    distributed random values.

    Args:
        dim (int): Dimension of the parameter vector.
        bounds (tuple): Lower and upper bounds for initialization values.
        tau (float): Optional tau parameter for self-adaptive mutation.

    Returns:
        Callable[[Pop], ParaVector]: Initializer function for population.
    """

    def initializer(_: Pop) -> ParaVector:
        vector = np.random.uniform(bounds[0], bounds[1], size=dim)
        return ParaVector(vector=vector, tau=tau)

    return initializer


def zero_initializer(dim: int) -> Callable[[Pop], ParaVector]:
    """
    Returns a factory function that creates ParaVector instances initialized with zeros.

    Args:
        dim (int): Dimension of the parameter vector.

    Returns:
        Callable[[Pop], ParaVector]: Initializer function for population.
    """

    def initializer(_: Pop) -> ParaVector:
        return ParaVector(vector=np.zeros(dim))

    return initializer


def fixed_initializer(
    values: np.ndarray, tau: float = 0.0
) -> Callable[[Pop], ParaVector]:
    """
    Returns a factory function that creates ParaVector instances with fixed predefined
    values.

    Args:
        values (np.ndarray): Vector to copy into each individual.
        tau (float): Optional tau parameter for self-adaptive mutation.

    Returns:
        Callable[[Pop], ParaVector]: Initializer function for population.
    """

    def initializer(_: Pop) -> ParaVector:
        return ParaVector(vector=values.copy(), tau=tau)

    return initializer


def normal_initializer(
    dim: int, mean: float = 0.0, std: float = 1.0, tau: float = 0.0
) -> Callable[[Pop], ParaVector]:
    """
    Returns a factory function that creates ParaVector instances initialized with
    normally distributed values.

    Args:
        dim (int): Dimension of the parameter vector.
        mean (float): Mean of the normal distribution.
        std (float): Standard deviation of the normal distribution.
        tau (float): Optional tau parameter for self-adaptive mutation.

    Returns:
        Callable[[Pop], ParaVector]: Initializer function for population.
    """

    def initializer(_: Pop) -> ParaVector:
        vector = np.random.normal(loc=mean, scale=std, size=dim)
        return ParaVector(vector=vector, tau=tau)

    return initializer
