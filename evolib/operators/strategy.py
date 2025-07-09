# SPDX-License-Identifier: MIT
"""
strategy.py – Core evolution loop strategies for evolutionary algorithms.

This module provides predefined evolution strategies such as (μ + λ) and (μ, λ)
in a modular form. Each function encapsulates one full generation cycle:
- offspring creation
- mutation
- fitness evaluation
- replacement
- statistics update

These functions assume that `Pop` has:
- a configured mutation strategy
- a registered fitness function via `set_functions()`

Functions:
- evolve_mu_plus_lambda: Classical (μ + λ) strategy with elitism.
- evolve_mu_comma_lambda: Classical (μ, λ) strategy without elitism.

All strategies are compatible with `strategy_registry` and `pop.run_one_generation()`.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evolib.core.population import Pop

from evolib.operators.mutation import mutate_offspring
from evolib.operators.replacement import (
    replace_generational,
    replace_steady_state,
    replace_truncation,
)
from evolib.operators.reproduction import generate_cloned_offspring


def evolve_mu_plus_lambda(pop: "Pop") -> None:
    """Elites and selected parents generate offspring, then mu best individuals are
    selected from parents + offspring."""

    if pop.fitness_function is None:
        raise ValueError(
            "No fitness function set in population."
            "Use pop.set_functions() before evolving."
        )
    if not pop.indivs:
        raise ValueError("Population is empty.")

    # Update mutation/crossover parameters
    pop.update_parameters()

    # CREATE OFFSPRING
    offspring = generate_cloned_offspring(pop.indivs, pop.offspring_pool_size)

    # OFFSPRING MUTATION
    mutate_offspring(pop, offspring)

    combined = pop.indivs + offspring

    # Evaluate fitness of all
    pop.evaluate_indivs(combined)

    # Select the best individuals
    replace_truncation(pop, combined)

    pop.update_statistics()


def evolve_mu_comma_lambda(pop: "Pop") -> None:
    """Parents generate offspring, but only offspring compete for the next generation
    (no elitism)."""

    if pop.fitness_function is None:
        raise ValueError(
            "No fitness function set in population."
            "Use pop.set_functions() before evolving."
        )
    if not pop.indivs:
        raise ValueError("Population is empty.")

    pop.evaluate_fitness()

    # Update mutation/crossover parameters
    pop.update_parameters()

    # CREATE OFFSPRING
    offspring = generate_cloned_offspring(pop.indivs, pop.offspring_pool_size)

    # OFFSPRING MUTATION
    mutate_offspring(pop, offspring)

    # Evaluate offspring fitness
    pop.evaluate_indivs(offspring)

    # REPLACE PARENTS
    replace_generational(pop, offspring)

    pop.update_statistics()


def evolve_steady_state(pop: "Pop") -> None:
    """
    Steady-State Evolution Strategy.

    In each generation, only a subset of individuals is replaced with offspring,
    while the rest of the population (including elites) is retained.

    Workflow:
    - Select parents
    - Generate offspring via cloning and crossover (if enabled)
    - Mutate offspring based on the current mutation strategy
    - Evaluate fitness of offspring
    - Replace the worst individuals (excluding elites) with offspring

    Notes:
    - The number of replaced individuals per generation is defined by `pop.lambda_`
    - Elites (top `pop.num_elites` individuals) are preserved
    - All individuals age automatically via `pop.update_statistics()`

    Raises:
        ValueError: If population is uninitialized or fitness function is missing
    """
    if pop.fitness_function is None:
        raise ValueError("No fitness function set. Use pop.set_functions() first.")
    if not pop.indivs:
        raise ValueError("Population is empty.")
    if pop.selection_fn is None:
        raise ValueError(
            "Selection strategy is required for steady_state evolution"
            "but not provided.\n"
            "Add e.g.:\n"
            "selection:\n"
            "  strategy: tournament\n"
            "to your YAML configuration."
        )

    # Select parents (configurable)
    parents = pop.select_parents(pop.lambda_)

    # Generate cloned offspring
    offspring = generate_cloned_offspring(parents, pop.lambda_)

    # Update mutation/crossover parameters
    pop.update_parameters()

    # Mutate offspring
    mutate_offspring(pop, offspring)

    # Evaluate offspring fitness
    pop.evaluate_indivs(offspring)

    # Replace worst individuals (excluding elites)
    replace_steady_state(pop, offspring, num_replace=pop.lambda_)

    # Update population statistics (including aging, diversity, fitness metrics)
    pop.update_statistics()


def evolve_flexible(pop: "Pop") -> None:
    """
    Modular evolution step using externally configured operators:
    - selection
    - crossover (optional)
    - mutation
    - replacement

    Assumes that `Pop` has:
    - a configured `selection_fn`
    - a valid `fitness_function`
    - a pre-built `_replacement_fn` (e.g., from ReplacementStrategy)
    """
    if pop.fitness_function is None:
        raise ValueError("No fitness function set in population.")
    if not pop.indivs:
        raise ValueError("Population is empty.")
    if pop.selection_fn is None:
        raise ValueError("Selection function not configured.")
    if pop._replacement_fn is None:
        raise ValueError("Replacement function not configured.")

    # Selection
    parents = pop.select_parents(pop.lambda_)

    # Reproduction
    offspring = generate_cloned_offspring(parents, pop.lambda_)

    # Mutation & Crossover Parameters
    pop.update_parameters()

    # Mutation
    mutate_offspring(pop, offspring)

    # Evaluate
    pop.evaluate_indivs(offspring)

    # Replacement (via configured strategy)
    pop._replacement_fn(pop, offspring)

    # Update statistics
    pop.update_statistics()
