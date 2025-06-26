"""
Example 02-01 - Step By Step Evolution


This example demonstrates the basic steps of evolutionary algorithms, including:

- Initializing a population with multiple individuals
- Applying mutation to the population
- Calculating fitness values before and after mutation
- Generating offspring and applying mutation
- Performing selection to retain the best individuals

Requirements:
- 'population.yaml' and 'individual.yaml' must be present in the current
working directory
"""

import random

from evolib import (
    Indiv,
    MutationParams,
    Pop,
    create_offspring_mu_lambda,
    mse_loss,
    mutate_gauss,
    mutate_offspring,
    simple_quadratic,
)


# User-defined fitness function
def my_fitness(indiv: Indiv) -> None:
    """Simple fitness function using the quadratic benchmark and MSE loss."""
    expected = 0.0
    predicted = simple_quadratic(indiv.para)
    indiv.fitness = mse_loss(expected, predicted)


# User-defined mutation function
def my_mutation(indiv: Indiv, params: MutationParams) -> None:
    """
    Mutates the individual's parameters using Gaussian mutation.

    Args:
        target_indiv (Indiv): The individual to mutate.
        params (MutationParams): Mutation context object that includes:
            - strength (float): Standard deviation of the Gaussian mutation.
            - bounds (tuple[float, float]): Value limits for the mutation.
            - rate (Optional[float]): (Unused) mutation rate, if applicable.
            - bias (Optional[float]): (Unused) bias mutation strength for
              neural networks.
    """
    indiv.para = mutate_gauss(indiv.para, params.strength, params.bounds)


# Load configuration and initialize population
pop = Pop(config_path="population.yaml")

for _ in range(pop.parent_pool_size):
    indiv = pop.create_indiv()
    indiv.para = random.uniform(-0.5, 0.5)  # Scalar parameter
    pop.add_indiv(indiv)

print("Parents:")
for i, indiv in enumerate(pop.indivs):
    my_fitness(indiv)
    print(f"  Indiv {i}: Parameter = {indiv.para:.4f}, Fitness = {indiv.fitness:.6f}")


# Generate Offspring
offspring = create_offspring_mu_lambda(pop.indivs, pop.offspring_pool_size)

# Evaluate fitness before mutation
print("\nOffspring before mutation:")
for i, indiv in enumerate(offspring):
    my_fitness(indiv)
    print(f"  Indiv {i}: Parameter = {indiv.para:.4f}, Fitness = {indiv.fitness:.6f}")

# Apply mutation
mutate_offspring(pop, offspring, my_mutation, bounds=(-1, 1))

# Evaluate fitness after mutation
print("\nOffspring after mutation:")
for i, indiv in enumerate(offspring):
    my_fitness(indiv)
    print(f"  Indiv {i}: Parameter = {indiv.para:.4f}, Fitness = {indiv.fitness:.6f}")


pop.indivs = pop.indivs + offspring
print("\nPopulation befor Selection")
for i, indiv in enumerate(pop.indivs):
    print(f"  Indiv {i}: Parameter = {indiv.para:.4f}, Fitness = {indiv.fitness:.6f}")

# Sort Population by fitness
pop.sort_by_fitness()

# Select best parents
pop.indivs = pop.indivs[: pop.parent_pool_size]

print("\nPopulation after Selection")
for i, indiv in enumerate(pop.indivs):
    print(f"  Indiv {i}: Parameter = {indiv.para:.4f}, Fitness = {indiv.fitness:.6f}")
