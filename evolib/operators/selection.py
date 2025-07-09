# SPDX-License-Identifier: MIT
"""
This module provides various parent selection strategies used in evolutionary
algorithms.

Included methods:
- Tournament Selection
- Rank-Based Selection (linear and exponential)
- Roulette Wheel Selection
- Stochastic Universal Sampling (SUS)
- Boltzmann (Softmax) Selection
- Truncation Selection

All methods operate on a population of individuals with assigned fitness values and
return deep copies of selected parents.
"""

import copy
import random
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np

from evolib.core.individual import Indiv

if TYPE_CHECKING:
    from evolib.core.population import Pop


def _calculate_rank_probabilities(
    ranks: np.ndarray, population_size: int, mode: str, exp_base: float
) -> np.ndarray:
    """
    Calculates selection probabilities based on ranks.

    Args:
        ranks (np.ndarray): Array of ranks (0 = worst, N-1 = best individual).
        population_size (int): Number of individuals in the population.
        mode (str): Selection mode, either 'linear' or 'exponential'.
        exp_base (float): Base used for exponential probability calculation.

    Returns:
        np.ndarray: Normalized selection probabilities summing to 1.

    Raises:
        ValueError: If mode is invalid or population_size < 1.
    """
    if population_size < 1:
        raise ValueError("Population size must be at least 1.")

    if mode == "linear":
        # Linear probability: p(i) = 2*(N - i) / (N*(N+1))
        probabilities = (2 * (population_size - ranks)) / (
            population_size * (population_size + 1)
        )
    elif mode == "exponential":
        # Exponential probability: p(i) = base^i / sum(base^j)
        probabilities = np.power(exp_base, ranks)
        probabilities = probabilities / np.sum(probabilities)
    else:
        raise ValueError("Selection mode must be either 'linear' or 'exponential'.")

    return probabilities


def selection_tournament(
    pop: "Pop",
    num_parents: int,
    tournament_size: int = 3,
    remove_selected: bool = False,
    fitness_maximization: bool = False,
) -> List[Indiv]:
    """
    Performs tournament selection to select parents from the population.

    Args:
        num_parents (int): Number of parents to select.
        tournament_size (int, optional): Number of individuals in each tournament.
                                      Defaults to 3.
        remove_selected (bool, optional): If True, selected individuals are removed
                                        from future tournaments. Defaults to False.
        minimize (bool, optional): If True, select individuals with lowest fitness
                                 (minimization). If False, select highest fitness
                                 (maximization). Defaults to True.

    Returns:
        list: List of selected parents.

    Raises:
        ValueError: If parameters are invalid (e.g., negative num_parents,
                   invalid tournament_size, or empty population).
    """

    # Input validation
    if not pop.indivs:
        raise ValueError("Population is empty")
    if num_parents < 0:
        raise ValueError("Number of parents must be non-negative")
    if tournament_size <= 0 or tournament_size > len(pop.indivs):
        raise ValueError(f"Tournament size must be between 1 and {len(pop.indivs)}")

    selected_parents = []
    available_indices = list(range(len(pop.indivs)))

    for _ in range(min(num_parents, len(pop.indivs))):
        if not available_indices:
            break

        # Zufaellige Auswahl von tournament_size Indizes
        tournament_indices = random.sample(
            available_indices, min(tournament_size, len(available_indices))
        )

        # Beste Fitness finden
        if fitness_maximization is False:
            best_idx, _ = min(
                ((i, pop.indivs[i].fitness) for i in tournament_indices),
                key=lambda x: x[1],
            )
        else:
            best_idx, _ = max(
                ((i, pop.indivs[i].fitness) for i in tournament_indices),
                key=lambda x: x[1],
            )

        # Kopie des besten Individuums hinzufuegen
        selected_parents.append(copy.deepcopy(pop.indivs[best_idx]))

        # Entfernen, falls gewuenscht
        if remove_selected:
            available_indices.remove(best_idx)

    return selected_parents


def selection_rank_based(
    pop: "Pop",
    *,
    num_parents: int,
    mode: str = "linear",
    remove_selected: bool = False,
    exp_base: float = 2.0,
    fitness_maximization: bool = False,
) -> List[Any]:
    """
    Performs rank-based selection using linear or exponential probability distribution.

    Selection probabilities are based on fitness ranks:
    - Linear:    p(i) = 2*(N-i)/(N*(N+1)), where i is the rank (0 = worst, N-1 = best).
    - Exponential: p(i) = base^i / sum(base^j), where base is a positive float.

    Args:
        num_parents (int): Number of parents to select.
        mode (str): Selection mode: 'linear' or 'exponential'. Default: 'linear'.
        remove_selected (bool): If True, selected individuals are removed from future
        selection. Default: False.
        exp_base (float): Base used in exponential probability calculation. Must be > 0.
        fitness_maximization (bool): If True, higher fitness is better.
        Default: False (minimization).

    Returns:
        List[Any]: List of selected individuals (deep copies).

    Raises:
        ValueError: If parameters are invalid (e.g., empty population,
        invalid base or mode).
        TypeError: If fitness values are invalid (None or NaN).
    """

    if not pop.indivs:
        raise ValueError("Population is empty")
    if not isinstance(num_parents, int) or num_parents < 0:
        raise ValueError("num_parents must be a non-negative integer")
    if remove_selected and num_parents > len(pop.indivs):
        raise ValueError(
            "num_parents cannot exceed population size when remove_selected=True"
        )
    if exp_base <= 0:
        raise ValueError("exp_base must be greater than 0")

    fitnesses = [indiv.fitness for indiv in pop.indivs]
    if any(f is None or np.isnan(f) for f in fitnesses):
        raise TypeError("All fitness values must be valid (not None or NaN)")

    # Sort individuals by fitness (ascending by default for minimization)
    sorted_indices = np.argsort(fitnesses)
    if fitness_maximization:
        sorted_indices = sorted_indices[::-1]  # Reverse for maximization

    selected_parents = []
    available_indices = sorted_indices.tolist()

    # Precompute probabilities if no removal
    precomputed_probabilities: Optional[np.ndarray] = None
    if not remove_selected:
        population_size = len(available_indices)
        ranks = np.arange(population_size)  # Rank 0 = worst, N-1 = best
        precomputed_probabilities = _calculate_rank_probabilities(
            ranks, population_size, mode, exp_base
        )

    for _ in range(num_parents):
        population_size = len(available_indices)
        if population_size == 0:
            raise ValueError("No individuals left for selection")

        # Compute probabilities
        if remove_selected:
            ranks = np.arange(population_size)
            probabilities = _calculate_rank_probabilities(
                ranks, population_size, mode, exp_base
            )
        else:
            assert precomputed_probabilities is not None
            probabilities = precomputed_probabilities

        # Select an individual based on rank probability
        list_index = np.random.choice(population_size, p=probabilities)
        selected_idx = available_indices[list_index]
        selected_parents.append(copy.deepcopy(pop.indivs[selected_idx]))

        # Optionally remove selected individual
        if remove_selected:
            available_indices.pop(list_index)

    return selected_parents


def selection_random(pop: "Pop", remove_selected: bool = False) -> List[Indiv]:
    """
    Performs random selection to select offspring from a population.

    Parameter:
        population: List of individuals, each with a fitness attribute.
        num_offspring: Number of offspring to select.

    return:
        List of selected individuals (offspring).
    """

    # Input validation
    # if tournament_size <= 0 or tournament_size > len(pop.indivs):
    #    raise ValueError(f"Tournament size must be between 1 and {len(pop.indivs)}")

    if pop.offspring_pool_size < 0:
        raise ValueError("Number of offspring must be non-negative")

    selected_parents = []
    available_indivs = pop.indivs.copy()

    for _ in range(min(pop.offspring_pool_size, len(pop.indivs))):
        if not available_indivs:
            print("Warning: available_indivs == 0")
            break

        # Randomly select individual from the population
        selected_parent = random.choice(available_indivs)
        parent = copy.deepcopy(selected_parent)

        # Append the best individual to offspring
        selected_parents.append(parent)

        if remove_selected:
            available_indivs.remove(parent)

    return selected_parents


def selection_roulette(
    pop: "Pop", num_parents: int, fitness_maximization: bool = False
) -> List[Any]:
    """
    Selects parents using fitness-proportional roulette wheel selection.

    Args:
        num_parents (int): Number of individuals to select.
        fitness_maximization (bool): If True, higher fitness is better.

    Returns:
        List[Any]: List of selected individuals (deep copies).
    """
    if not pop.indivs:
        raise ValueError("Population is empty")

    fitnesses = np.array([indiv.fitness for indiv in pop.indivs])
    if any(np.isnan(fitnesses)) or any(f is None for f in fitnesses):
        raise TypeError("Invalid fitness values")

    if not fitness_maximization:
        max_fitness = np.max(fitnesses)
        fitnesses = max_fitness - fitnesses + 1e-12  # Prevent zero or negative values

    total_fitness = np.sum(fitnesses)
    if total_fitness == 0:
        raise ValueError("Total fitness is zero; selection is undefined")

    probabilities = fitnesses / total_fitness
    indices = np.random.choice(len(pop.indivs), size=num_parents, p=probabilities)

    return [copy.deepcopy(pop.indivs[i]) for i in indices]


def selection_sus(
    pop: "Pop", num_parents: int, fitness_maximization: bool = False
) -> List[Any]:
    """
    Selects individuals using Stochastic Universal Sampling (SUS).

    Args:
        num_parents (int): Number of individuals to select.
        fitness_maximization (bool): If True, higher fitness is better.

    Returns:
        List[Any]: Selected individuals (deep copies).
    """
    if not pop.indivs:
        raise ValueError("Population is empty")

    fitnesses = np.array([ind.fitness for ind in pop.indivs])
    if any(np.isnan(fitnesses)) or any(f is None for f in fitnesses):
        raise TypeError("Invalid fitness values")

    if not fitness_maximization:
        max_fitness = np.max(fitnesses)
        fitnesses = max_fitness - fitnesses + 1e-12

    total_fitness = np.sum(fitnesses)
    if total_fitness == 0:
        raise ValueError("Total fitness is zero; selection is undefined")

    probabilities = fitnesses / total_fitness
    cumulative = np.cumsum(probabilities)

    step = 1.0 / num_parents
    start = np.random.uniform(0, step)
    pointers = [start + i * step for i in range(num_parents)]

    selected = []
    i = 0
    for p in pointers:
        while p > cumulative[i]:
            i += 1
        selected.append(copy.deepcopy(pop.indivs[i]))

    return selected


def selection_boltzmann(
    pop: "Pop",
    num_parents: int,
    temperature: float = 1.0,
    fitness_maximization: bool = False,
) -> List[Any]:
    """
    Selects individuals using Boltzmann (Softmax) selection.

    Args:
        num_parents (int): Number of individuals to select.
        temperature (float): Controls selection pressure (higher = more uniform).
        fitness_maximization (bool): If True, higher fitness is better.

    Returns:
        List[Any]: Selected individuals (deep copies).
    """
    if not pop.indivs:
        raise ValueError("Population is empty")
    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero")

    fitnesses = np.array([ind.fitness for ind in pop.indivs])
    if any(np.isnan(fitnesses)) or any(f is None for f in fitnesses):
        raise TypeError("Invalid fitness values")

    fitnesses = np.array(fitnesses, dtype=np.float64)

    if fitness_maximization:
        scaled = fitnesses / temperature
    else:
        scaled = -fitnesses / temperature

    # Softmax
    exp_values = np.exp(scaled - np.max(scaled))  # for numerical stability
    probabilities = exp_values / np.sum(exp_values)

    indices = np.random.choice(len(pop.indivs), size=num_parents, p=probabilities)
    return [copy.deepcopy(pop.indivs[i]) for i in indices]


def selection_truncation(
    pop: "Pop", num_parents: int, fitness_maximization: bool = False
) -> List[Any]:
    """
    Selects the top individuals (by fitness) deterministically.

    Args:
        num_parents (int): Number of individuals to select.
        fitness_maximization (bool): If True, selects best fitness; else lowest.

    Returns:
        List[Any]: Selected individuals (deep copies).
    """
    if not pop.indivs:
        raise ValueError("Population is empty")
    if num_parents > len(pop.indivs):
        raise ValueError("Cannot select more parents than population size")

    fitnesses = [ind.fitness for ind in pop.indivs]
    if any(f is None or np.isnan(f) for f in fitnesses):
        raise TypeError("Invalid fitness values")

    sorted_indices = np.argsort(fitnesses)
    if fitness_maximization:
        sorted_indices = sorted_indices[::-1]

    selected_indices = sorted_indices[:num_parents]
    return [copy.deepcopy(pop.indivs[i]) for i in selected_indices]
