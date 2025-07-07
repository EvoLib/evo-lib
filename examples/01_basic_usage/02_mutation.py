"""
Example 01-02 - Mutation

This example demonstrates:
- How to create a population using configuration files.
- How to apply mutation using the `mutate` interface of EvoLib.
- How parameter values change as a result of mutation.
"""

from evolib import Pop

# Load example configuration for the population
# Uses the mutation strategy defined in population.yaml (e.g. constant or exponential)
pop = Pop(config_path="population.yaml")

# Create a single individual
indiv = pop.create_indiv()

# Show parameter before mutation
print(f"Before mutation: {indiv.para.get_status()}")

# Apply mutation
indiv.mutate()

# Show parameter after mutation
print(f"After mutation:  {indiv.para.get_status()}")
