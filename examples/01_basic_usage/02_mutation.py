"""
Example 01-02 - Mutation

This example demonstrates:
- How to create an individual and a population using configuration files.
- How to define a custom mutation function.
- How to apply mutation using the `mutate` interface of EvoLib.
- How parameter values change as a result of mutation.
"""

import random

from evolib import Indiv, MutationParams, Pop, mutate_gauss, mutate_indiv


# User-defined mutation function passed to mutate_offspring().
# This allows full flexibility in how mutations are applied.
# Here, we use Gaussian mutation on scalar parameters.
# It must match the signature:
#   (indiv: Indiv, strength: float, bounds: tuple) -> None
def my_mutation(indiv: Indiv, params: MutationParams) -> None:
    """
    Simple Gaussian mutation for demonstration purposes.

    Args:
        indiv (Indiv): The individual to mutate.
        params (MutationParams): Mutation context object that includes:
            - strength (float): Standard deviation of the Gaussian mutation.
            - bounds (tuple[float, float]): Value limits for the mutation.
            - rate (Optional[float]): (Unused) mutation rate, if applicable.
            - bias (Optional[float]): (Unused) bias mutation strength for
              neural networks.
    """
    # Apply Gaussian mutation to parameter
    mutated = mutate_gauss(indiv.para, params.strength, bounds=params.bounds)
    indiv.para = mutated


# Load example configuration for the population
pop = Pop(config_path="population.yaml", mutation_function=my_mutation)

# Create a single individual
my_indiv = pop.create_indiv()

# Set initial parameter (scalar in this simple example)
my_indiv.para = random.uniform(-0.5, 0.5)

# Show parameter before mutation
print(f"Before mutation: {my_indiv.para:.4f}")

# Apply mutation using the explicitly passed function (for clarity).
# We could also use pop.mutation_function, but passing my_mutation
# directly makes the example more transparent and self-contained.
mutate_indiv(pop, my_indiv, mutation_function=my_mutation, bounds=(-1, 1))

# Show parameter after mutation
print(f"After mutation:  {my_indiv.para:.4f}")
