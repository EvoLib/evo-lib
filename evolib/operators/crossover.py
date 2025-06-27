# SPDX-License-Identifier: MIT
"""
Collection of crossover operators for evolutionary algorithms.

Includes implementations of arithmetic, blend (BLX-Alpha), simulated binary (SBX),
intermediate, heuristic, and differential crossover. Designed for use with real-valued
vectors and adaptable to various evolutionary strategies.
"""

import random
from typing import Callable, List, Tuple, Union

import numpy as np


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
    crossover_probability: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Simulated Binary Crossover (SBX) on two parent vectors.

    SBX creates offspring that simulate the effect of single-point binary crossover
    in real-valued search spaces, controlled by a distribution index η.

    Args:
        parent1 (np.ndarray): First parent vector.
        parent2 (np.ndarray): Second parent vector.
        eta (float): Distribution index (controls spread; higher = closer to parents).
        crossover_probability (float): Probability to apply crossover. If not applied,
        parents are copied.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two offspring vectors.

    Raises:
        ValueError: If parent vectors have different lengths.
    """
    parent1 = np.array(parent1)
    parent2 = np.array(parent2)

    if len(parent1) != len(parent2):
        raise ValueError("Parent vectors must have the same length.")

    if random.random() >= crossover_probability:
        return parent1.copy(), parent2.copy()

    child1 = np.zeros_like(parent1)
    child2 = np.zeros_like(parent1)

    for idx, _ in enumerate(parent1):
        if random.random() < 0.5:
            u = random.random()
            if u <= 0.5:
                beta = (2 * u) ** (1 / (eta + 1))
            else:
                beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))

            x1 = parent1[idx]
            x2 = parent2[idx]
            child1[idx] = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
            child2[idx] = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
        else:
            child1[idx] = parent1[idx]
            child2[idx] = parent2[idx]

    return child1, child2


def crossover_intermediate(
    parent1: np.ndarray,
    parent2: np.ndarray,
    d: float = 0.25,
    crossover_probability: float = 0.7,
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
        Crossover_rate (float): Probability of performing crossover.
        If not applied, parents are copied.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two offspring vectors.

    Raises:
        ValueError: If parent vectors have different lengths.
    """

    parent1 = np.array(parent1)
    parent2 = np.array(parent2)

    if len(parent1) != len(parent2):
        raise ValueError("Parent vectors must have the same length.")

    if random.random() >= crossover_probability:
        return parent1.copy(), parent2.copy()

    child1 = np.zeros_like(parent1)
    child2 = np.zeros_like(parent1)

    for idx, parent_s in enumerate(parent1):
        alpha1 = random.uniform(-d, 1 + d)
        alpha2 = random.uniform(-d, 1 + d)
        diff = parent2[idx] - parent_s
        child1[idx] = parent1[idx] + alpha1 * diff
        child2[idx] = parent1[idx] + alpha2 * diff

    return child1, child2


def crossover_heuristic(
    parent1: np.ndarray,
    parent2: np.ndarray,
    fitness_func: Callable[[np.ndarray], float],
    crossover_probability: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform heuristic crossover on two parent vectors.

    A new offspring is created by extrapolating from the worse parent
    toward the better one, guided by their fitness values.

    Args:
        parent1 (np.ndarray): First parent vector.
        parent2 (np.ndarray): Second parent vector.
        fitness_func (Callable): Fitness evaluation function. Should
        return lower values for better individuals.
        crossover_probability (float): Probability of performing crossover.
        Otherwise, parents are returned unchanged.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A new offspring vector and the better
        parent (as second child).

    Raises:
        ValueError: If parent vectors have different lengths.
    """
    parent1 = np.array(parent1)
    parent2 = np.array(parent2)

    if len(parent1) != len(parent2):
        raise ValueError("Parent vectors must have the same length.")

    if random.random() >= crossover_probability:
        return parent1.copy(), parent2.copy()

    # Ensure parent2 is the better (lower fitness) individual
    if fitness_func(parent1) < fitness_func(parent2):
        parent1, parent2 = parent2, parent1

    alpha = random.random()  # Degree of extrapolation toward better parent
    child = parent1 + alpha * (parent2 - parent1)

    return child, parent2.copy()


def crossover_differential(
    parent1: np.ndarray, population: List[np.ndarray], F: float = 0.5, CR: float = 0.9
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Differential Evolution crossover (DE/rand/1/bin).

    A mutant vector is generated from three random individuals, and then
    recombined with the target vector (parent1) using binomial crossover.

    Args:
        parent1 (np.ndarray): The target vector (individual to be perturbed).
        population (List[np.ndarray]): The current population, must include at
        least 4 distinct individuals.
        F (float): Differential weight (scaling factor for mutation).
        Typically in [0.4, 1.0].
        CR (float): Crossover probability (controls how much of the mutant is used).

    Returns:
        Tuple[np.ndarray, np.ndarray]: A new offspring and a backup (copy of parent1).

    Raises:
        ValueError: If population is too small or contains duplicate individuals.
    """
    parent1 = np.array(parent1)
    n = len(parent1)

    # Ensure sufficient distinct individuals
    others = [ind for ind in population if not np.array_equal(ind, parent1)]
    if len(others) < 3:
        raise ValueError("Population must contain at least 4 distinct individuals.")

    r1, r2, r3 = map(np.array, random.sample(others, 3))
    mutant = r1 + F * (r2 - r3)

    # Binomial crossover
    child = parent1.copy()
    j_rand = random.randint(0, n - 1)
    for i in range(n):
        if random.random() < CR or i == j_rand:
            child[i] = mutant[i]

    return child, parent1.copy()
