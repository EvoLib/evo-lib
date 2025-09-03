# SPDX-License-Identifier: MIT
"""
Collection of crossover operators for evolutionary algorithms.

Includes implementations of arithmetic, blend (BLX-Alpha), simulated binary (SBX),
intermediate, heuristic, and differential crossover. Designed for use with real-valued
vectors and adaptable to various evolutionary strategies.
"""

import random
from typing import TYPE_CHECKING, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from evolib.core.individual import Indiv
    from evolib.core.population import Pop


def crossover_blend_alpha(
    parent1_para: np.ndarray,
    parent2_para: np.ndarray,
    alpha: float = 0.5,
    num_children: int = 2,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Perform Blend-Alpha Crossover (BLX-Alpha) on two parent vectors.

    This operator creates one or two offspring by sampling each gene
    from an extended interval around the parent genes.

    Args:
        parent1_para (np.ndarray): Parameter vector of the first parent.
        parent2_para (np.ndarray): Parameter vector of the second parent.
        alpha (float): Expansion factor for the sampling interval. Default is 0.5.
        num_children (int): Number of children to generate (1 or 2). Default is 2.

    Returns:
        np.ndarray or tuple of np.ndarray: One or two offspring vectors.

    Raises:
        ValueError: If parent vectors have different lengths or `num_children`
        is invalid.
    """
    parent1_para = np.array(parent1_para)
    parent2_para = np.array(parent2_para)

    if len(parent1_para) != len(parent2_para):
        raise ValueError("Parent vectors must have the same length.")
    if num_children not in [1, 2]:
        raise ValueError("num_children must be 1 or 2.")

    def generate_child() -> np.ndarray:
        child = np.zeros_like(parent1_para)
        for idx, parent1 in enumerate(parent1_para):
            min_val = min(parent1, parent2_para[idx])
            max_val = max(parent1, parent2_para[idx])
            delta = max_val - min_val
            lower = min_val - alpha * delta
            upper = max_val + alpha * delta
            child[idx] = random.uniform(lower, upper)
        return child

    child1 = generate_child()
    if num_children == 1:
        return child1
    child2 = generate_child()
    return child1, child2


def crossover_arithmetic(
    parent1_para: np.ndarray, parent2_para: np.ndarray, num_children: int = 2
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Perform arithmetic crossover between two parent vectors.

    Each child's genes are a weighted average of the corresponding genes
    of the parents, using a randomly chosen mixing coefficient alpha ∈ [0, 1].

    Args:
        parent1_para (np.ndarray): Parameter vector of the first parent.
        parent2_para (np.ndarray): Parameter vector of the second parent.
        num_children (int): Number of children to return (1 or 2). Default is 2.

    Returns:
        np.ndarray or tuple of np.ndarray: One or two offspring vectors.

    Raises:
        ValueError: If parent vectors have different lengths or num_children is invalid.
    """
    parent1_para = np.array(parent1_para)
    parent2_para = np.array(parent2_para)

    if len(parent1_para) != len(parent2_para):
        raise ValueError("Parent vectors must have the same length.")
    if num_children not in [1, 2]:
        raise ValueError("num_children must be 1 or 2.")

    alpha = random.random()
    child1 = alpha * parent1_para + (1 - alpha) * parent2_para

    if num_children == 1:
        return child1

    child2 = (1 - alpha) * parent1_para + alpha * parent2_para
    return child1, child2


def crossover_simulated_binary(
    parent1: np.ndarray,
    parent2: np.ndarray,
    eta: float = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Simulated Binary Crossover (SBX) on two parent vectors.

    SBX creates offspring that simulate the effect of single-point binary crossover
    in real-valued search spaces, controlled by a distribution index η.

    Args:
        parent1 (np.ndarray): First parent vector.
        parent2 (np.ndarray): Second parent vector.
        eta (float): Distribution index (controls spread; higher = closer to parents).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two offspring vectors.

    Raises:
        ValueError: If parent vectors have different lengths.
    """
    parent1 = np.array(parent1)
    parent2 = np.array(parent2)

    if len(parent1) != len(parent2):
        raise ValueError("Parent vectors must have the same length.")

    child1 = np.zeros_like(parent1)
    child2 = np.zeros_like(parent1)

    for idx, _ in enumerate(parent1):
        u = random.random()
        if u <= 0.5:
            beta = (2 * u) ** (1 / (eta + 1))
        else:
            beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))

        x1 = parent1[idx]
        x2 = parent2[idx]
        child1[idx] = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
        child2[idx] = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)

    return child1, child2


def crossover_intermediate(
    parent1: np.ndarray,
    parent2: np.ndarray,
    blend_range: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform intermediate crossover with extended alpha range on two parent vectors.

    Each offspring gene is calculated using a random alpha ∈ [-d, 1 + d],
    creating solutions inside and outside the segment between parents.

    Args:
        parent1 (np.ndarray): First parent vector.
        parent2 (np.ndarray): Second parent vector.
        d (float): Extension factor for sampling interval beyond [0, 1].
        Default is 0.25.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two offspring vectors.

    Raises:
        ValueError: If parent vectors have different lengths.
    """

    parent1 = np.array(parent1)
    parent2 = np.array(parent2)

    if len(parent1) != len(parent2):
        raise ValueError("Parent vectors must have the same length.")

    child1 = np.zeros_like(parent1)
    child2 = np.zeros_like(parent1)

    for idx, parent_s in enumerate(parent1):
        alpha1 = random.uniform(-blend_range, 1 + blend_range)
        alpha2 = random.uniform(-blend_range, 1 + blend_range)
        diff = parent2[idx] - parent_s
        child1[idx] = parent1[idx] + alpha1 * diff
        child2[idx] = parent1[idx] + alpha2 * diff
    return child1, child2


def crossover_offspring(pop: "Pop", offspring: list["Indiv"]) -> None:
    """
    Perform crossover for each pair of offspring individuals. Supports ParaComposite by
    applying module-specific crossover operators and probabilities defined in the
    configuration.

    Notes:
        - Offspring are assumed to be copied before this call.
        - Individuals are paired (0,1), (2,3), ...
        - This method does not return; offspring are modified in place.
    """

    if not offspring:
        return

    for i in range(0, len(offspring) - 1, 2):
        child1 = offspring[i]
        child2 = offspring[i + 1]

        for module_name in child1.para.components:
            module1 = child1.para.components[module_name]
            module2 = child2.para.components[module_name]

            prob1 = module1.evo_params.crossover_probability
            prob2 = module2.evo_params.crossover_probability

            if prob1 is None or prob2 is None:
                continue

            effective_prob = min(prob1, prob2)

            if effective_prob <= 0.0 or np.random.rand() >= effective_prob:
                continue

            # Perform crossover
            module1.crossover_with(module2)
