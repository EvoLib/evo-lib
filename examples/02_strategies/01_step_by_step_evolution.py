"""
Example 02-01 - Step By Step Evolution


This example demonstrates the basic steps of evolutionary algorithms, including:

- Initializing a population with multiple individuals
- Applying mutation to the population
- Calculating fitness values before and after mutation
- Generating offspring and applying mutation
- Performing selection to retain the best individuals

Requirements:
    'population.yaml' must be present in the current working directory
"""

from evolib import Indiv, Pop, generate_cloned_offspring, mse_loss, simple_quadratic


# User-defined fitness function
def my_fitness(indiv: Indiv) -> None:
    """Simple fitness function using the quadratic benchmark and MSE loss."""
    expected = 0.0
    predicted = simple_quadratic(indiv.para.vector)
    indiv.fitness = mse_loss(expected, predicted)


# Create and initialize the population (default behavior).
pop = Pop(config_path="population.yaml")

# Advanced usage:
# You can disable automatic initialization if needed (e.g., for testing or
# custom setups).
# pop = Pop(config_path="01_config.yaml", initialize=False)
# pop.initialize_population()

print("Parents:")
for i, indiv in enumerate(pop.indivs):
    my_fitness(indiv)
    print(
        f"  Indiv {i}: Parameter = {indiv.para.vector}, Fitness = {indiv.fitness:.6f}"
    )


# Generate Offspring
offspring = generate_cloned_offspring(pop.indivs, pop.offspring_pool_size)

# Evaluate fitness before mutation
print("\nOffspring before mutation:")
for i, indiv in enumerate(offspring):
    my_fitness(indiv)
    print(
        f"  Indiv {i}: Parameter = {indiv.para.vector}, Fitness = {indiv.fitness:.6f}"
    )

# Apply mutation
for indiv in offspring:
    indiv.mutate()

# Evaluate fitness after mutation
print("\nOffspring after mutation:")
for i, indiv in enumerate(offspring):
    my_fitness(indiv)
    print(
        f"  Indiv {i}: Parameter = {indiv.para.vector}, Fitness = {indiv.fitness:.6f}"
    )


pop.indivs = pop.indivs + offspring
print("\nPopulation befor Selection")
for i, indiv in enumerate(pop.indivs):
    print(
        f"  Indiv {i}: Parameter = {indiv.para.vector}, Fitness = {indiv.fitness:.6f}"
    )

# Sort Population by fitness
pop.sort_by_fitness()

# Select best parents
pop.indivs = pop.indivs[: pop.parent_pool_size]

print("\nPopulation after Selection")
for i, indiv in enumerate(pop.indivs):
    print(
        f"  Indiv {i}: Parameter = {indiv.para.vector}, Fitness = {indiv.fitness:.6f}"
    )
