"""
Example 01-02 - Mutation

This example demonstrates:
- How to create an individual and a population using configuration files.
- How to define a custom mutation function.
- How to apply mutation using the `mutate` interface of EvoLib.
- How parameter values change as a result of mutation.
"""

import random

from evolib import Indiv, Pop


# Load example configuration for the population
pop = Pop(config_path="population.yaml")

# Create a single individual
my_indiv = pop.create_indiv()

# Show parameter before mutation
print(f"Before mutation: {my_indiv.para}")

# Apply mutation
my_indiv.mutate()

# Show parameter after mutation
print(f"After mutation:  {my_indiv.para}")
