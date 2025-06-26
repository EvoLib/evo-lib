# SPDX-License-Identifier: MIT

from typing import Any, List

import numpy as np

from evolib.globals.numeric import MAX_FLOAT
from evolib.interfaces.enums import Origin
from evolib.utils.copy_indiv import copy_indiv


def create_offspring_mu_lambda(parents: List[Any], lambda_: int) -> List[Any]:
    """
    Generates offspring individuals by randomly selecting parents (with replacement).

    Used in (μ, λ) evolution strategies.

    Args:
        parents (List[Any]): List of selected parent individuals.
        lambda_ (int): Number of offspring to generate.

    Returns:
        List[Any]: List of offspring individuals.
    """
    if not parents:
        raise ValueError("parents cannot be empty")
    if lambda_ <= 0:
        raise ValueError("lambda_ must be greater than zero")

    offspring = []
    parent_indices = np.random.choice(len(parents), size=lambda_, replace=True)

    for idx in parent_indices:
        parent = parents[idx]
        child = copy_indiv(parent)
        child.age = 0
        child.fitness = MAX_FLOAT
        child.origin = Origin.OFFSPRING
        child.parent_idx = idx
        offspring.append(child)

    return offspring
