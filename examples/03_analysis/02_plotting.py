"""
Example 03-02 - Plotting

This example shows how to visualize the evolution history collected during a run.
It demonstrates how to:

- Access history data from the population
- Plot fitness statistics over generations
- Interpret trends using matplotlib
"""

from evolib import Indiv, Pop, mse_loss, simple_quadratic
from evolib.utils.plotting import plot_fitness


def my_fitness(indiv: Indiv) -> None:
    expected = 0.0
    predicted = simple_quadratic(indiv.para.vector)
    indiv.fitness = mse_loss(expected, predicted)


# Setup
pop = Pop(config_path="population.yaml")
pop.set_functions(fitness_function=my_fitness)

# Evolution
for _ in range(pop.max_generations):
    pop.run_one_generation()

# History to DataFrame
history = pop.history_logger.to_dataframe()

# Plotting
plot_fitness(history, show=True, save_path="./figures/02_plotting.png")
print("Plot saved to ./figures/02_plotting.png.")
