"""
Example 01-03 - Population Mutation

This example demonstrates how to:
- Initialize a population with multiple individuals
- Apply mutation to all individuals
- Observe how mutation affects each parameter
"""

import random

from evolib import Indiv, MutationParams, Pop, mutate_gauss, mutate_offspring


# User-defined mutation function passed to mutate_offspring().
# This allows full flexibility in how mutations are applied.
# Here, we use Gaussian mutation on scalar parameters.
# It must match the signature:
#   (indiv: Indiv, strength: float, bounds: tuple) -> None
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
    mutated = mutate_gauss(indiv.para, params.strength, params.bounds)
    indiv.para = mutated


# Load configuration for the population (mutation rate, size, etc.)
pop = Pop(config_path="population.yaml")

# Load configuration for a single individual (mutation strength, etc.)

# Create and initialize individuals
for _ in range(pop.parent_pool_size):
    new_indiv = pop.create_indiv()
    new_indiv.para = random.uniform(-0.5, 0.5)  # simple scalar parameter
    pop.add_indiv(new_indiv)

# Print parameters before mutation
print("Before mutation:")
for i, indiv in enumerate(pop.indivs):
    print(f"  Indiv {i}: {indiv.para:.4f}")

# Apply mutation to all individuals
mutate_offspring(pop, pop.indivs, mutation_function=my_mutation, bounds=(-1, 1))

# Print parameters after mutation
print("\nAfter mutation:")
for i, indiv in enumerate(pop.indivs):
    print(f"  Indiv {i}: {indiv.para:.4f}")
