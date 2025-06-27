"""
Example 02-06 – Adaptive Individual Mutation with Tau.

This example demonstrates how each individual maintains its own adaptive mutation
parameters, including a dynamic `tau` value. The strategy used is
`adaptive_individual`, enabling fine-grained evolution of mutation behavior.

Key Elements:
- Each individual gets its own tau based on parameter length.
- Mutation strength is adapted individually
- Fitness is computed using the Rosenbrock benchmark.
"""

import random

import numpy as np

from evolib import (
    Indiv,
    MutationParams,
    Pop,
    Strategy,
    adapt_mutation_strength,
    default_update_tau,
    evolve_mu_lambda,
    mse_loss,
    mutate_gauss,
    rosenbrock,
)


# Fitness function using Rosenbrock
def my_fitness(indiv: Indiv) -> None:
    expected = [1.0, 1.0, 1.0, 1.0]
    predicted = rosenbrock(indiv.para)
    indiv.fitness = mse_loss(expected, predicted)


# Mutation function with per-individual tau-based strength adaptation
def my_mutation(indiv: Indiv, params: MutationParams) -> None:
    # Mutate mutation_strength using tau
    adapt_mutation_strength(indiv, params)

    # Apply Gaussian mutation
    for i, val in enumerate(indiv.para):
        indiv.para[i] = mutate_gauss(val, indiv.mutation_strength, params.bounds)


# Initialization with per-individual parameters and tau
def initialize_population(pop: Pop) -> None:
    for _ in range(pop.parent_pool_size):
        new_indiv = pop.create_indiv()
        new_indiv.para = np.array([-2.0, 2.0, -1.5, 1.5])
        # random starting value
        new_indiv.mutation_strength = random.uniform(
            pop.min_mutation_strength, pop.max_mutation_strength
        )
        new_indiv.mutation_probability = 1.0
        pop.tau_update_function(new_indiv)  # computes tau based on length of para
        pop.add_indiv(new_indiv)

    for indiv in pop.indivs:
        my_fitness(indiv)


# Run experiment
def run_experiment(config_path: str) -> None:
    pop = Pop(config_path)
    pop.set_functions(my_fitness, my_mutation, tau_update_function=default_update_tau)
    initialize_population(pop)

    for _ in range(pop.max_generations):
        evolve_mu_lambda(pop, my_fitness, my_mutation, strategy=Strategy.MU_PLUS_LAMBDA)
        pop.print_status(verbosity=1)
        mean_mutation_strengths = np.mean(
            np.array([indiv.mutation_strength for indiv in pop.indivs])
        )
        print(f"Mean mutation strength: {mean_mutation_strengths:0.5f}\n")


print("Running adaptive_individual experiment with tau...\n")
run_experiment("06_adaptive_individual.yaml")
