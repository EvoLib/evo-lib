"""
Example 01-04 - Fitness

This example demonstrates how to:
- Initialize a population with multiple individuals
- Apply mutation to all individuals
- Observe how mutation affects parameters and fitness
- Evaluate fitness before and after mutation
"""

from evolib import Indiv, Pop, mse_loss, simple_quadratic


# User-defined fitness function that is passed to the evolution loop.
# This function should assign a fitness value to the given individual.
# Here, we use a simple benchmark (quadratic function) and the MSE loss.
def my_fitness(indiv: Indiv) -> None:
    """
    Simple fitness function using the quadratic benchmark and MSE loss.

    Assigns fitness based on distance to 0.0 (global minimum).
    """
    expected = 0.0
    predicted = simple_quadratic(indiv.para.vector)
    indiv.fitness = mse_loss(expected, predicted)


# Load configuration and initialize population
pop = Pop(config_path="04_fitness.yaml")

for _ in range(pop.parent_pool_size):
    indiv = pop.create_indiv()
    pop.add_indiv(indiv)

# Evaluate fitness before mutation
print("Before mutation:")
for i, indiv in enumerate(pop.indivs):
    my_fitness(indiv)
    print(
        f"  Indiv {i}: Parameter = {indiv.para.vector}, Fitness = {indiv.fitness:.6f}"
    )

# Apply mutation
for indiv in pop.indivs:
    indiv.mutate()

# Evaluate fitness after mutation
print("\nAfter mutation:")
for i, indiv in enumerate(pop.indivs):
    my_fitness(indiv)
    print(
        f"  Indiv {i}: Parameter = {indiv.para.vector}, Fitness = {indiv.fitness:.6f}"
    )
