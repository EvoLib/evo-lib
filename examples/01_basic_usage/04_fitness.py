"""
Example 01-04 - Fitness

This example demonstrates how to:
- Initialize a population with multiple individuals
- Apply mutation to all individuals
- Observe how mutation affects parameters and fitness
- Evaluate fitness before and after mutation
"""

import random

from evolib import (
    Indiv,
    MutationParams,
    Pop,
    mse_loss,
    mutate_gauss,
    mutate_offspring,
    simple_quadratic,
)


# User-defined fitness function that is passed to the evolution loop.
# This function should assign a fitness value to the given individual.
# Here, we use a simple benchmark (quadratic function) and the MSE loss.
def my_fitness(indiv: Indiv) -> None:
    """
    Simple fitness function using the quadratic benchmark and MSE loss.

    Args:
        indiv (Indiv): The individual to evaluate. The function should
        assign the computed fitness to `target_indiv.fitness`.
        params (MutationParams): Mutation context object that includes:
            - strength (float): Standard deviation of the Gaussian mutation.
            - bounds (tuple[float, float]): Value limits for the mutation.
            - rate (Optional[float]): (Unused) mutation rate, if applicable.
            - bias (Optional[float]): (Unused) bias mutation strength for
              neural networks.
    """
    expected = 0.0
    predicted = simple_quadratic(indiv.para)
    indiv.fitness = mse_loss(expected, predicted)


# User-defined mutation function passed to mutate_offspring().
# This allows full flexibility in how mutations are applied.
# Here, we use Gaussian mutation on scalar parameters.
def my_mutation(indiv: Indiv, params: MutationParams) -> None:
    """
    Mutates the individual's parameters using Gaussian mutation.

    Args:
        target_indiv (Indiv): The individual to mutate.
        mutation_strength (float): The mutation strength.
        bounds (tuple): Value bounds (min, max) for mutation.
    """
    indiv.para = mutate_gauss(indiv.para, params.strength, params.bounds)


# Load configuration and initialize population
pop = Pop(config_path="population.yaml")

for _ in range(pop.parent_pool_size):
    indiv = pop.create_indiv()
    indiv.para = random.uniform(-0.5, 0.5)  # Scalar parameter
    pop.add_indiv(indiv)

# Evaluate fitness before mutation
print("Before mutation:")
for i, indiv in enumerate(pop.indivs):
    my_fitness(indiv)
    print(f"  Indiv {i}: Parameter = {indiv.para:.4f}, Fitness = {indiv.fitness:.6f}")

# Apply mutation
mutate_offspring(pop, pop.indivs, my_mutation, bounds=(-1, 1))

# Evaluate fitness after mutation
print("\nAfter mutation:")
for i, indiv in enumerate(pop.indivs):
    my_fitness(indiv)
    print(f"  Indiv {i}: Parameter = {indiv.para:.4f}, Fitness = {indiv.fitness:.6f}")
