# SPDX-License-Identifier: MIT

from typing import Any, List

import numpy as np


def generate_cloned_offspring(parents: List[Any], lambda_: int) -> List[Any]:
    """
    Creates ``lambda_`` cloned offspring by randomly selecting parents with replacement.

    Each offspring is a deep copy of a randomly chosen parent.
    This method performs no crossover or mutation and is typically used in evolutionary
    strategies such as (mu, ``lambda``), (mu + ``lambda``), or steady-state evolution to
    initialize raw offspring before variation operators are applied.

    Args:
        parents (List[Any]): List of parent individuals to clone from.
        ``lambda_`` (int): Number of offspring to create.

    Returns:
        List[Any]: List of cloned offspring individuals.

    Raises:
        ValueError: If the parent list is empty or ``lambda_`` is not positive.
    """
    if not parents:
        raise ValueError("parents cannot be empty")
    if lambda_ <= 0:
        raise ValueError("lambda_ must be greater than zero")

    offspring = []
    parent_indices = np.random.choice(len(parents), size=lambda_, replace=True)

    for idx in parent_indices:
        parent = parents[idx]
        child = parent.copy(
            reset_id=True,
            reset_fitness=True,
            reset_age=True,
            reset_evaluation=True,
            reset_origin=True,
        )
        child.parent_idx = idx
        offspring.append(child)

    return offspring
