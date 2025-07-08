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
from evolib.operators.replacement import replace_generational, replace_truncation
from evolib.operators.reproduction import create_offspring_mu_lambda


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

    # Update mutation parameters
    pop.update_mutation_parameters()

    # CREATE OFFSPRING
    offspring = create_offspring_mu_lambda(pop.indivs, pop.offspring_pool_size)

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

    # Update mutation parameters
    pop.update_mutation_parameters()

    # CREATE OFFSPRING
    offspring = create_offspring_mu_lambda(pop.indivs, pop.offspring_pool_size)

    # OFFSPRING MUTATION
    mutate_offspring(pop, offspring)

    # Evaluate offspring fitness
    pop.evaluate_indivs(offspring)

    # REPLACE PARENTS
    replace_generational(pop, offspring)

    pop.update_statistics()
