# SPDX-License-Identifier: MIT

from typing import Callable

from evolib.core.population import Indiv, Pop
from evolib.interfaces.enums import Strategy
from evolib.interfaces.types import MutationFunction
from evolib.operators.mutation import mutate_offspring
from evolib.operators.replacement import replace_generational, replace_truncation
from evolib.operators.reproduction import create_offspring_mu_lambda


def evolve_mu_lambda(
    pop: Pop,
    fitness_function: Callable[[Indiv], None],
    mutation_function: MutationFunction,
    *,
    strategy: Strategy = Strategy.MU_PLUS_LAMBDA,
    bounds: tuple[float, float] = (-5, 5),
    max_age: int = 0,
    update_stats: bool = True,
) -> None:
    """
    Performs one evolutionary step using either (μ+λ) or (μ,λ) evolution strategy.

    Args:
        pop (Pop): The current population.
        fitness_function (Callable): Function to evaluate the fitness of an individual.
        mutation_function (Callable): Function to mutate an individual.
        strategy (Strategy): The strategy to use: MU_PLUS_LAMBDA or MU_COMMA_LAMBDA.
        bounds (tuple): Tuple specifying the mutation bounds (min, max).
        max_age (int): Maximum allowed age for individuals (used in MU_COMMA_LAMBDA).
        update_stats (bool): Whether to update the population statistics.

    Raises:
        ValueError: If the strategy is invalid or the population is empty.
    """

    if strategy not in [Strategy.MU_PLUS_LAMBDA, Strategy.MU_COMMA_LAMBDA]:
        raise ValueError(
            "Invalid strategy. Use Strategy.MU_PLUS_LAMBDA or Strategy.MU_COMMA_LAMBDA."
        )
    if not pop.indivs:
        raise ValueError("Population is empty.")

    # Optional: Evaluate parents if elites are to be kept in comma strategy
    if strategy == Strategy.MU_COMMA_LAMBDA and pop.num_elites > 0:
        for indiv in pop.indivs:
            fitness_function(indiv)

    # CREATE OFFSPRING
    offspring = create_offspring_mu_lambda(pop.indivs, pop.offspring_pool_size)

    # OFFSPRING MUTATION
    mutate_offspring(pop, offspring, mutation_function, bounds=bounds)

    if strategy == Strategy.MU_PLUS_LAMBDA:
        combined = pop.indivs + offspring

        # Evaluate fitness of all
        for indiv in combined:
            fitness_function(indiv)

        # Select the best individuals
        replace_truncation(pop, combined)
    else:  # MU_COMMA_LAMBDA
        # Evaluate offspring fitness
        for indiv in offspring:
            fitness_function(indiv)

        # REPLACE PARENTS
        replace_generational(pop, offspring, max_age=max_age)

    if update_stats:
        pop.update_statistics()
