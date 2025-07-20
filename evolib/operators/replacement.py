# SPDX-License-Identifier: MIT

import random
from typing import TYPE_CHECKING, List

import numpy as np

if TYPE_CHECKING:
    from evolib.core.population import Pop

from evolib.core.population import Indiv
from evolib.interfaces.enums import Origin
from evolib.utils.fitness import sort_by_fitness


def replace_truncation(
    pop: "Pop", offspring: List[Indiv], fitness_maximization: bool = False
) -> None:
    """
    Replacement via truncation: select the top μ individuals from a given list
    (e.g., offspring or parents + offspring) based on fitness.

    This strategy performs a pure replacement without elitism, aging,
    or additional constraints.

    Suitable for:
    - (μ, λ) strategies
    - (μ + λ) strategies without elitism
    - educational purposes (clean, minimal logic)

    See also: `replace_generational` for elitism and aging support.

    Args:
        pop (Pop): The population object whose individuals will be replaced.
        offspring (List[Indiv]): A list of newly generated offspring individuals.
        fitness_maximization: If True, higher fitness is better.
    """

    if not offspring:
        raise ValueError("Offspring list must not be empty.")
    if len(offspring) < pop.parent_pool_size:
        raise ValueError("Not enough offspring to fill the parent pool.")

    # Sort offspring by fitness in ascending order (assuming lower is better)
    sorted_offspring = sort_by_fitness(offspring, maximize=fitness_maximization)

    # Keep current elites from previous generation
    elites = pop.get_elites()

    # Select remaining individuals
    remaining = sorted_offspring[: pop.parent_pool_size - len(elites)]

    # Combine elites + top offspring
    pop.indivs = elites + remaining


def replace_mu_lambda(
    pop: "Pop", offspring: List[Indiv], fitness_maximization: bool = False
) -> None:
    """
    Replaces the population with new offspring using the mu+lambda strategy, followed by
    resetting the parent index and origin of the individuals.

    Args:
        pop (Pop): The population object whose individuals will be replaced.
        offspring (List[Indiv]): A list of newly generated offspring individuals.
    """

    replace_truncation(pop, offspring, fitness_maximization)

    for indiv in pop.indivs:
        indiv.parent_idx = None
        indiv.origin = Origin.PARENT


def replace_generational(
    pop: "Pop",
    offspring: List[Indiv],
    max_age: int = 0,
    fitness_maximization: bool = False,
) -> None:
    """
    Replace the population with offspring, preserving elites and optionally applying
    age-based filtering. Resulting population is sorted by fitness.

    This function implements generational replacement with elitism and
    optional aging. The final population size will be at most pop.parent_pool_size.

    Args:
        pop (Pop): The population object.
        offspring (List[Indiv]): Newly generated offspring.
        max_age (int): Maximum allowed individual age (0 = disabled).
        fitness_maximization (bool): If True, higher fitness is better.

    Raises:
        ValueError: On invalid configuration or population state.
    """
    if not offspring:
        raise ValueError("Offspring list cannot be empty.")
    if pop.num_elites < 0:
        raise ValueError(f"num_elites ({pop.num_elites}) cannot be negative.")
    if pop.num_elites > len(pop.indivs):
        raise ValueError(
            f"num_elites ({pop.num_elites}) cannot exceed population size "
            f"({len(pop.indivs)})."
        )
    if max_age < 0:
        raise ValueError("max_age must be ≥ 0.")

    # Sort and mark elites
    elites = pop.get_elites()

    # Combine offspring with current population (needed for aging step)
    combined = elites + offspring

    # Filter by age if aging is active
    if max_age > 0:
        survivors = [
            indiv for indiv in combined if indiv.is_elite or indiv.age < max_age
        ]
    else:
        survivors = combined

    # Sort by fitness (best first)
    sorted_survivors = sort_by_fitness(survivors, maximize=fitness_maximization)

    # Truncate to desired population size
    pop.indivs = sorted_survivors[: pop.parent_pool_size]


def replace_steady_state(
    pop: "Pop",
    offspring: List[Indiv],
    num_replace: int = 0,
    fitness_maximization: bool = False,
) -> None:
    """
    Replace the worst individuals in the population with offspring, preserving elite
    individuals. Implements steady-state replacement.

    Args:
        pop (Pop): Current population.
        offspring (List[Indiv]): New individuals to insert.
        num_replace (int): Number of individuals to replace.
            If 0, replaces len(offspring).
        fitness_maximization (bool): Whether higher fitness is better.

    Raises:
        ValueError: If replacement configuration is invalid.
    """
    if not offspring:
        raise ValueError("Offspring list cannot be empty.")

    if num_replace is None or num_replace <= 0:
        num_replace = len(offspring)

    if num_replace > len(pop.indivs):
        raise ValueError(
            f"num_replace ({num_replace}) cannot exceed "
            f"population size ({len(pop.indivs)})."
        )
    if num_replace > len(offspring):
        raise ValueError(
            f"num_replace ({num_replace}) cannot exceed number of "
            f"offspring ({len(offspring)})."
        )
    if pop.num_elites < 0:
        raise ValueError(f"num_elites ({pop.num_elites}) cannot be negative.")
    if pop.num_elites > len(pop.indivs):
        raise ValueError(
            f"num_elites ({pop.num_elites}) cannot exceed "
            f"population size ({len(pop.indivs)})."
        )

    # Get sorted elite individuals and mark them
    elites = pop.get_elites()

    # Define replaceable pool (non-elites only)
    non_elites = [indiv for indiv in pop.indivs if not indiv.is_elite]

    if num_replace > len(non_elites):
        raise ValueError(
            f"Not enough non-elites ({len(non_elites)}) to "
            f"replace {num_replace} individuals."
        )

    # Sort non-elites by fitness (worst at the end)
    sorted_non_elites = sort_by_fitness(non_elites, maximize=fitness_maximization)

    # Replace worst non-elites with best offspring
    sorted_offspring = sort_by_fitness(offspring, maximize=fitness_maximization)

    survivors = (
        elites + sorted_non_elites[:-num_replace] + sorted_offspring[:num_replace]
    )

    # Final sort for consistency
    survivors = sort_by_fitness(survivors, maximize=fitness_maximization)
    pop.indivs = survivors


def replace_random(pop: "Pop", offspring: List[Indiv]) -> None:
    """
    Replace random non-elite individuals in the population with new offspring.

    Elites are preserved using `pop.get_elites()` and marked via `is_elite = True`.

    Args:
        pop (Pop): The population object.
        offspring (List[Indiv]): New offspring individuals.

    Raises:
        ValueError: If offspring is empty or replacement is not possible.
    """
    if not offspring:
        raise ValueError("Offspring list cannot be empty.")
    if pop.num_elites < 0:
        raise ValueError(f"num_elites ({pop.num_elites}) cannot be negative.")
    if pop.num_elites > len(pop.indivs):
        raise ValueError(
            f"num_elites ({pop.num_elites}) cannot exceed "
            f"population size ({len(pop.indivs)})."
        )

    # Retrieve and mark elites
    elites = pop.get_elites()

    # Determine non-elite pool
    non_elites = [indiv for indiv in pop.indivs if not indiv.is_elite]

    if len(offspring) > len(non_elites):
        raise ValueError(
            f"Not enough non-elites ({len(non_elites)}) to "
            f"replace {len(offspring)} individuals."
        )

    # Select random replacement positions in non-elite pool
    replace_indices = random.sample(range(len(non_elites)), len(offspring))

    # Apply replacement
    for i, idx in enumerate(replace_indices):
        non_elites[idx] = offspring[i]

    # Combine and sort final population
    pop.indivs = elites + non_elites


def replace_weighted_stochastic(
    pop: "Pop",
    offspring: List[Indiv],
    temperature: float = 1.0,
    fitness_maximization: bool = False,
) -> None:
    """
    Replace individuals in the population using inverse-fitness-weighted softmax
    sampling, preserving elite individuals.

    Args:
        pop (Pop): The population object.
        offspring (List[Indiv]): List of new individuals.
        temperature (float): Softmax temperature (> 0).
        fitness_maximization (bool): Whether higher fitness is better.

    Raises:
        ValueError: On invalid input or if not enough non-elites are available.
    """
    if not offspring:
        raise ValueError("Offspring list cannot be empty.")
    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")
    if pop.num_elites < 0:
        raise ValueError(f"num_elites ({pop.num_elites}) cannot be negative.")
    if pop.num_elites > len(pop.indivs):
        raise ValueError(
            f"num_elites ({pop.num_elites}) cannot exceed population size "
            f"({len(pop.indivs)})."
        )

    # Retrieve and mark elites
    elites = pop.get_elites()

    # Determine non-elite individuals
    non_elites = [indiv for indiv in pop.indivs if not indiv.is_elite]

    if len(offspring) > len(non_elites):
        raise ValueError(
            f"Cannot replace {len(offspring)} individuals; only {len(non_elites)} "
            "non-elites available."
        )

    # Extract fitness values from non-elites
    fitness = np.array([indiv.fitness for indiv in non_elites], dtype=np.float64)

    # Compute inverse-scaled softmax probabilities
    if not fitness_maximization:
        scaled = -fitness / temperature
    else:
        scaled = fitness / temperature

    exp_scores = np.exp(scaled - np.max(scaled))  # numerical stability
    probabilities = exp_scores / np.sum(exp_scores)

    # Sample unique indices in non-elites for replacement
    replace_indices = np.random.choice(
        len(non_elites), size=len(offspring), replace=False, p=probabilities
    )

    # Replace selected individuals
    for i, idx in enumerate(replace_indices):
        non_elites[idx] = offspring[i]

    # Recombine and sort
    combined = elites + non_elites
    combined = sort_by_fitness(combined, maximize=fitness_maximization)
    pop.indivs = combined
