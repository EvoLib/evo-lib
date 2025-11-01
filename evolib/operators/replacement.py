# SPDX-License-Identifier: MIT

import random
from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    from evolib.core.population import Pop

from evolib.core.population import Indiv
from evolib.interfaces.enums import Origin
from evolib.utils.fitness import sort_by_fitness
from evolib.utils.lineage_logger import LineageLogger


def _mark_removed_indivs(
    all_candidates: list["Indiv"],
    survivors: list["Indiv"],
    generation: int,
    lineage_logger: Optional["LineageLogger"] = None,
) -> None:
    """Mark individuals removed from population and optionally log them."""
    removed = []
    for indiv in all_candidates:
        if indiv not in survivors:
            indiv.exit_gen = generation
            removed.append(indiv)

    # Log removal events immediately
    if lineage_logger is not None and removed:
        lineage_logger.log_population(
            removed, generation, event="died", note="removed in replacement"
        )


def replace_truncation(
    pop: "Pop", pool: List[Indiv], fitness_maximization: bool = False
) -> None:
    """
    Generic truncation replacement selecting the top μ individuals from a given pool.
    Pass a pool that already contains what should compete (e.g., parents+offspring for
    μ+λ, or offspring only for μ,λ).

    Args:
        pop: Population handle (μ taken from pop.parent_pool_size).
        pool: Candidate list to pick survivors from.
        fitness_maximization: If True, higher fitness is better.
    """
    if not pool:
        raise ValueError("Pool must not be empty.")
    if pop.parent_pool_size <= 0:
        raise ValueError("Parent pool size (mu) must be positive.")
    if len(pool) < pop.parent_pool_size:
        raise ValueError("Pool smaller than parent_pool_size; cannot truncate cleanly.")

    sorted_pool = sort_by_fitness(pool, maximize=fitness_maximization)
    survivors = sorted_pool[: pop.parent_pool_size]

    # Deduplicate and mark removed
    all_candidates = list({x.id: x for x in (pop.indivs + pool)}.values())
    _mark_removed_indivs(
        all_candidates, survivors, pop.generation_num, pop.lineage_logger
    )

    pop.indivs = survivors
    for indiv in pop.indivs:
        indiv.origin = Origin.PARENT


def replace_mu_plus_lambda(
    pop: "Pop", offspring: List[Indiv], fitness_maximization: bool = False
) -> None:
    """
    (μ + λ) replacement: parents and offspring compete together; keep best μ.

    Precondition:
        - Fitness of *both* parents and offspring has been evaluated
          (same environmental context).

    Effect:
        - No explicit 'elitism' branch needed; if a parent is truly elite,
          it simply ranks among the top μ and survives.

    Args:
        pop: Population object (parents in pop.indivs).
        offspring: Newly generated offspring.
        fitness_maximization: Whether fitness is to be maximized.
    """
    if not offspring:
        raise ValueError("Offspring list must not be empty.")

    combined = pop.indivs + offspring
    replace_truncation(pop, combined, fitness_maximization)


def replace_mu_comma_lambda(
    pop: "Pop",
    offspring: list[Indiv],
    fitness_maximization: bool = False,
) -> None:
    """
    (mu, lambda) replacement: ONLY offspring compete; keep best mu offspring, top
    `num_elites` parents are preserved. Parents do not compete and cannot survive solely
    due to elitism.

    Precondition:
        - Fitness of offspring has been evaluated.
        - Parents may have fitness values, but they are not considered here.

    Args:
        pop: Population object (current parents are in pop.indivs).
        offspring: Newly generated and evaluated offspring.
        fitness_maximization: Whether fitness is to be maximized.
    """

    if not offspring:
        raise ValueError("Offspring list must not be empty.")
    if pop.num_elites < 0:
        raise ValueError("num_elites cannot be negative.")
    if pop.num_elites > pop.parent_pool_size:
        raise ValueError("num_elites cannot exceed μ.")

    # Keep elites from parents
    elites = pop.get_elites() if pop.num_elites > 0 else []

    # Mark all non-elites as removed (parents that die this gen)
    old_non_elites = [ind for ind in pop.indivs if ind not in elites]
    _mark_removed_indivs(old_non_elites, [], pop.generation_num, pop.lineage_logger)

    # Select best (mu - num_elites) offspring
    sorted_offspring = sort_by_fitness(offspring, maximize=fitness_maximization)
    survivors = elites + sorted_offspring[: pop.parent_pool_size - len(elites)]

    # Mark any remaining offspring that were not chosen as removed
    _mark_removed_indivs(offspring, survivors, pop.generation_num, pop.lineage_logger)

    # Replace population
    pop.indivs = survivors
    for indiv in pop.indivs:
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

    # Mark old parents and non-selected offspring as removed
    all_candidates = pop.indivs + offspring
    _mark_removed_indivs(
        all_candidates, survivors, pop.generation_num, pop.lineage_logger
    )

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

    # Mark old parents and non-selected offspring as removed
    all_candidates = pop.indivs + offspring
    _mark_removed_indivs(
        all_candidates, survivors, pop.generation_num, pop.lineage_logger
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

    survivors = elites + non_elites

    # Mark old parents and non-selected offspring as removed
    all_candidates = pop.indivs + offspring
    _mark_removed_indivs(
        all_candidates, survivors, pop.generation_num, pop.lineage_logger
    )

    # Combine and sort final population
    pop.indivs = survivors


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

    survivors = elites + non_elites

    # Mark old parents and non-selected offspring as removed
    all_candidates = pop.indivs + offspring
    _mark_removed_indivs(
        all_candidates, survivors, pop.generation_num, pop.lineage_logger
    )

    # Recombine and sort
    survivors = sort_by_fitness(survivors, maximize=fitness_maximization)
    pop.indivs = survivors
