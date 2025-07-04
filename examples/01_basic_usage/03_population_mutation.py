"""
Example 01-03 - Population Mutation

This example demonstrates how to:
- Initialize a population with multiple individuals
- Apply mutation to all individuals
- Observe how mutation affects each parameter
"""

from evolib import Pop

# Load configuration for the population (mutation rate, size, etc.)
pop = Pop(config_path="population.yaml")

# Create and initialize individuals
for _ in range(pop.parent_pool_size):
    my_indiv = pop.create_indiv()
    pop.add_indiv(my_indiv)

# Print parameters before mutation
print("Before mutation:")
for i, indiv in enumerate(pop.indivs):
    print(f"  Indiv {i}: {indiv.para.get_status()}")

# Apply mutation to all individuals
for indiv in pop.indivs:
    indiv.mutate()

# Print parameters after mutation
print("\nAfter mutation:")
for i, indiv in enumerate(pop.indivs):
    print(f"  Indiv {i}: {indiv.para.get_status()}")
