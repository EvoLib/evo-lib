# SPDX-License-Identifier: MIT

from typing import Any, List

import numpy as np

from evolib.globals.numeric import MAX_FLOAT
from evolib.interfaces.enums import Origin
from evolib.utils.copy_indiv import copy_indiv


def generate_cloned_offspring(parents: List[Any], lambda_: int) -> List[Any]:
    """
    Creates lambda_ cloned offspring by randomly selecting parents with replacement.

    Each offspring is a deep copy of a randomly chosen parent, with age reset and
    fitness set to MAX_FLOAT. This method performs no crossover or mutation and
    is typically used in evolutionary strategies such as (mu, lambda), (mu + lambda),
    or steady-state evolution to initialize raw offspring before variation operators
    are applied.

    Args:
        parents (List[Any]): List of parent individuals to clone from.
        lambda_ (int): Number of offspring to create.

    Returns:
        List[Any]: List of cloned offspring individuals.

    Raises:
        ValueError: If the parent list is empty or lambda_ is not positive.
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
