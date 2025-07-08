# SPDX-License-Identifier: MIT

import random
from typing import TYPE_CHECKING, List

import numpy as np

if TYPE_CHECKING:
    from evolib.core.population import Pop

from evolib.core.population import Indiv
from evolib.interfaces.enums import Origin


def replace_truncation(pop: "Pop", offspring: List[Indiv]) -> None:
    """
    Replaces the current population with the fittest individuals from the offspring
    using truncation selection.

    Args:
        pop (Pop): The population object whose individuals will be replaced.
        offspring (List[Indiv]): A list of newly generated offspring individuals.
    """

    if not offspring:
        raise ValueError("Offspring list must not be empty.")
    if len(offspring) < pop.parent_pool_size:
        raise ValueError("Not enough offspring to fill the parent pool.")

    # Sort offspring by fitness in ascending order (assuming lower is better)
    sorted_offspring = sorted(offspring, key=lambda indiv: indiv.fitness)

    # Truncate to the desired number of parents
    pop.indivs = sorted_offspring[: pop.parent_pool_size]


def replace_mu_lambda(pop: "Pop", offspring: List[Indiv]) -> None:
    """
    Replaces the population with new offspring using the mu+lambda strategy, followed by
    resetting the parent index and origin of the individuals.

    Args:
        pop (Pop): The population object whose individuals will be replaced.
        offspring (List[Indiv]): A list of newly generated offspring individuals.
    """

    replace_truncation(pop, offspring)

    for indiv in pop.indivs:
        indiv.parent_idx = None
        indiv.origin = Origin.PARENT


def replace_generational(pop: "Pop", offspring: List[Indiv], max_age: int = 0) -> None:
    """
    Replace the population with new offspring, optionally preserving elite individuals.
    Can mimic steady-state if num_elite is high.

    Args:
        pop (Pop): The population object whose individuals will be replaced.
        offspring (List[Indiv]): A list of new offspring individuals.
        max_age (int, optional): Maximum age for individuals. If set, individuals
        older than this age will be removed.
        Defaults to 0 (no age limit).

    Raises:
        ValueError: If offspring is empty, num_elite is negative, or there are
                    insufficient offspring
                    to fill the population after preserving elites.
    """

    if not offspring:
        raise ValueError("Offspring list cannot be empty")
    if pop.num_elites < 0:
        raise ValueError(f"num_elite ({pop.num_elites}) cannot be negative")
    if pop.num_elites > len(pop.indivs):
        raise ValueError(
            f"num_elite ({pop.num_elites}) cannot exceed population"
            f"size ({len(pop.indivs)})"
        )
    if max_age < 0:
        raise ValueError("max_age must be greater than 0")

    # Calculate required offspring count after preserving elites
    required_offspring = len(pop.indivs) - pop.num_elites
    if required_offspring > len(offspring) and max_age == 0:
        raise ValueError(
            f"Insufficient offspring ({len(offspring)}) to fill population "
            f"({len(pop.indivs)}) with {pop.num_elites} elites"
        )

    # Sort population and offspring by fitness (ascending)
    pop.sort_by_fitness()
    offspring.sort(key=lambda indivs: indivs.fitness)

    if max_age > 0:
        pop.indivs.extend(offspring)
        for indiv in pop.indivs[pop.num_elites :]:
            if indiv.age >= max_age:
                pop.indivs.remove(indiv)

    else:
        # Replace the population with offspring, keeping elites
        pop.indivs[pop.num_elites :] = offspring[:required_offspring]


def replace_steady_state(
    pop: "Pop", offspring: List[Indiv], num_replace: int = 0
) -> None:
    """
    Replace the worst individuals in the population with new offspring using steady-
    state strategy. Can mimic generational replacement if num_replace equals the
    population size.

    Args:
        pop (Pop): The current population.
        offspring (List[Indiv]): List of new individuals to iinsert into the population.
        num_replace (int, optional): Number of individuals to replace.
        Defaults to len(offspring).

    Raises:
        ValueError: If num_replace exceeds population size or is negative, or if i
        offspring is empty.
    """
    if not offspring:
        raise ValueError("Offspring list cannot be empty")

    if num_replace is None:
        num_replace = len(offspring)

    if num_replace <= 0:
        raise ValueError("num_replace must be greater than 0.")
    if num_replace > len(pop.indivs):
        raise ValueError(
            f"num_replace ({num_replace}) cannot exceed population"
            f"size ({len(pop.indivs)})."
        )
    if num_replace > len(offspring):
        raise ValueError(
            f"num_replace ({num_replace}) cannot exceed number of"
            f"offspring ({len(offspring)})."
        )

    # Sort population by ascending fitness
    pop.sort_by_fitness()

    # Sort offspring by ascending fitness
    sorted_offspring = sorted(offspring, key=lambda indiv: indiv.fitness)

    # Replace the worst individuals with new offspring
    pop.indivs[-num_replace:] = sorted_offspring[:num_replace]


def replace_random(pop: "Pop", offspring: List[Indiv]) -> None:
    """
    Replace random individuals in the population with new offspring. Optionally
    preserving elite individuals.

    Args:
        offspring (list): List of new individuals to add to the population.

    Raises:
        ValueError: If offspring is empty, num_elites is negative, or there are
        insufficient offspring to fill the population after preserving elites.

    Returns:
        None (modifies pop.indivs in place).
    """

    if not offspring:
        raise ValueError("Offspring list cannot be empty")
    if pop.num_elites < 0:
        raise ValueError(f"num_elite ({pop.num_elites}) cannot be negative")
    if pop.num_elites > len(pop.indivs):
        raise ValueError(
            f"num_elite ({pop.num_elites}) cannot exceed population"
            f"size ({len(pop.indivs)})"
        )

    max_replaceable = len(pop.indivs) - pop.num_elites
    if len(offspring) > max_replaceable:
        raise ValueError(
            f"Offspring size ({len(offspring)}) exceeds replaceable individuals "
            f"({max_replaceable}) after preserving {pop.num_elites} elites"
        )

    #  Sort population by fitness (ascending)
    pop.sort_by_fitness()

    # Wähle zufällige Indizes für die Ersetzung (nur nicht-elitäre Individuen)
    replace_indices = random.sample(
        range(pop.num_elites, len(pop.indivs)), len(offspring)
    )

    # Ersetze die ausgewählten Individuen mit offspring
    for i, idx in enumerate(replace_indices):
        pop.indivs[idx] = offspring[i]


def replace_weighted_stochastic(
    pop: "Pop", offspring: List[Indiv], temperature: float = 1.0
) -> None:
    """
    Replaces individuals in the population with offspring, using a probability weighted
    by inverse fitness (lower fitness = more likely to survive).

    Args:
        pop (Pop): The population object.
        offspring (List[Indiv]): List of new individuals.
        temperature (float): Softness of selection (higher = more random).
        Default is 1.0.
    """
    if not offspring:
        raise ValueError("Offspring list cannot be empty.")
    if len(offspring) > len(pop.indivs):
        raise ValueError("Offspring size cannot exceed population size.")

    # Fitness-Array (niedrig besser)
    fitness = np.array([indiv.fitness for indiv in pop.indivs])

    # Normiere mit Softmax über negatives Fitnessmaß
    # (damit schlechte höhere Wahrscheinlichkeit haben)
    inverse_scaled = -fitness / temperature
    probabilities = np.exp(
        inverse_scaled - np.max(inverse_scaled)
    )  # Für numerische Stabilität
    probabilities /= np.sum(probabilities)

    # Wähle Indizes aus der Population, die ersetzt werden
    replace_indices = np.random.choice(
        len(pop.indivs), size=len(offspring), replace=False, p=probabilities
    )

    # Ersetze ausgewählte Individuen
    for i, idx in enumerate(replace_indices):
        pop.indivs[idx] = offspring[i]
