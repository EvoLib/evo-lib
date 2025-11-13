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

from evolib.operators.crossover import crossover_offspring
from evolib.operators.heli import run_heli
from evolib.operators.mutation import mutate_offspring
from evolib.operators.replacement import (
    replace_mu_comma_lambda,
    replace_mu_plus_lambda,
    replace_steady_state,
)
from evolib.operators.reproduction import generate_cloned_offspring


def evolve_mu_plus_lambda(pop: "Pop") -> None:
    """Elites and selected parents generate offspring, then mu best individuals are
    selected from parents + offspring."""

    if pop.fitness_function is None:
        raise ValueError("No fitness function set in population.")
    if not pop.indivs:
        raise ValueError("Population is empty.")

    # Age all current individuals
    pop.age_indivs()

    # CREATE OFFSPRING
    offspring = generate_cloned_offspring(
        pop.indivs, pop.offspring_pool_size, current_gen=pop.generation_num
    )

    # Lineage Logging
    if pop.lineage_logger is not None:
        pop.lineage_logger.log_population(offspring, pop.generation_num, event="born")

    # Update mutation/crossover parameters
    pop.update_parameters(offspring)

    # Crossover
    crossover_offspring(pop, offspring)

    # OFFSPRING MUTATION
    mutate_offspring(pop, offspring)

    # Optional HELI
    pop.heli_fitness_evaluations_gen = 0
    if pop.heli_enabled is True:
        pop.heli_fitness_evaluations_gen = run_heli(pop, offspring)
        pop.heli_fitness_evaluations_total += pop.heli_fitness_evaluations_gen

    combined = pop.indivs + offspring

    # Evaluate fitness of all
    pop.evaluate_indivs(combined)

    # Select the best individuals
    replace_mu_plus_lambda(pop, offspring)

    # Remove individuals that exceed max_age
    pop.remove_old_indivs()

    # Lineage Logging
    if pop.lineage_logger is not None:
        pop.lineage_logger.log_population(
            pop.indivs, pop.generation_num, event="survived"
        )

    pop.update_statistics()


def evolve_mu_comma_lambda(pop: "Pop") -> None:
    """
    Only offspring compete for survival; parents are replaced.

    If `num_elites > 0`, top elite parents are preserved and their fitness is re-
    evaluated before offspring generation.
    """

    if pop.fitness_function is None:
        raise ValueError(
            "No fitness function set in population."
            "Use pop.set_functions() before evolving."
        )
    if not pop.indivs:
        raise ValueError("Population is empty.")

    # Age all current individuals
    pop.age_indivs()

    # Evaluate current parents only if elites will be preserved
    if pop.num_elites > 0:
        pop.evaluate_fitness()

    # CREATE OFFSPRING
    offspring = generate_cloned_offspring(
        pop.indivs, pop.offspring_pool_size, current_gen=pop.generation_num
    )

    # Lineage Logging
    if pop.lineage_logger is not None:
        pop.lineage_logger.log_population(offspring, pop.generation_num, event="born")

    # Update mutation/crossover parameters
    pop.update_parameters(offspring)

    # Crossover
    crossover_offspring(pop, offspring)

    # OFFSPRING MUTATION
    mutate_offspring(pop, offspring)

    # Optional HELI
    pop.heli_fitness_evaluations_gen = 0
    if pop.heli_enabled is True:
        pop.heli_fitness_evaluations_gen = run_heli(pop, offspring)
        pop.heli_fitness_evaluations_total += pop.heli_fitness_evaluations_gen

    # Evaluate offspring fitness
    pop.evaluate_indivs(offspring)

    # REPLACE PARENTS
    replace_mu_comma_lambda(pop, offspring)

    # Lineage Logging
    if pop.lineage_logger is not None:
        pop.lineage_logger.log_population(
            pop.indivs, pop.generation_num, event="survived"
        )

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

    # Age all current individuals
    pop.age_indivs()

    # Select parents (configurable)
    if (
        pop.config.selection is not None
        and pop.config.selection.num_parents is not None
    ):
        num_parents = pop.config.selection.num_parents
    else:
        num_parents = pop.offspring_pool_size
    parents = pop.select_parents(num_parents)

    # Generate cloned offspring
    offspring = generate_cloned_offspring(
        parents, pop.lambda_, current_gen=pop.generation_num
    )

    # Lineage Logging
    if pop.lineage_logger is not None:
        pop.lineage_logger.log_population(offspring, pop.generation_num, event="born")

    # Update mutation/crossover parameters
    pop.update_parameters(offspring)

    # Crossover
    crossover_offspring(pop, offspring)

    # Mutate offspring
    mutate_offspring(pop, offspring)

    # Optional HELI
    pop.heli_fitness_evaluations_gen = 0
    if pop.heli_enabled is True:
        pop.heli_fitness_evaluations_gen = run_heli(pop, offspring)
        pop.heli_fitness_evaluations_total += pop.heli_fitness_evaluations_gen

    # Evaluate offspring fitness
    pop.evaluate_indivs(offspring)

    # Replace worst individuals (excluding elites)
    replace_steady_state(pop, offspring, num_replace=pop.lambda_)

    # Remove individuals that exceed max_age
    pop.remove_old_indivs()

    # Lineage Logging
    if pop.lineage_logger is not None:
        pop.lineage_logger.log_population(
            pop.indivs, pop.generation_num, event="survived"
        )

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

    # Age all current individuals
    pop.age_indivs()

    # Fitness
    pop.evaluate_fitness()

    # Selection
    if (
        pop.config.selection is not None
        and pop.config.selection.num_parents is not None
    ):
        num_parents = pop.config.selection.num_parents
    else:
        num_parents = pop.offspring_pool_size
    parents = pop.select_parents(num_parents)

    # Reproduction
    offspring = generate_cloned_offspring(
        parents, pop.lambda_, current_gen=pop.generation_num
    )

    # Lineage Logging
    if pop.lineage_logger is not None:
        pop.lineage_logger.log_population(offspring, pop.generation_num, event="born")

    # Mutation & Crossover Parameters
    pop.update_parameters(offspring)

    # Crossover
    crossover_offspring(pop, offspring)

    # Mutation
    mutate_offspring(pop, offspring)

    # Evaluate
    pop.evaluate_indivs(offspring)

    # Replacement (via configured strategy)
    pop._replacement_fn(pop, offspring)

    # Remove individuals that exceed max_age
    pop.remove_old_indivs()

    # Lineage Logging
    if pop.lineage_logger is not None:
        pop.lineage_logger.log_population(
            pop.indivs, pop.generation_num, event="survived"
        )

    # Update statistics
    pop.update_statistics()
