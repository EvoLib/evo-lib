"""
Example 03-02 - Plotting

This example shows how to visualize the evolution history collected during a run.
It demonstrates how to:

- Access history data from the population
- Plot fitness statistics over generations
"""

from evolib import Indiv, Pop, mse_loss, simple_quadratic
from evolib.utils.plotting import plot_fitness, plot_history


def my_fitness(indiv: Indiv) -> None:
    expected = 0.0
    predicted = simple_quadratic(indiv.para["test-vector"].vector)
    indiv.fitness = mse_loss(expected, predicted)


# Setup
pop = Pop(config_path="population.yaml", fitness_function=my_fitness)

# Run evolution
pop.run(verbosity=0)

# 1. Basic plotting (default metrics: best, mean, and median fitness)
plot_fitness(pop, show=True, save_path="./figures/02_plotting.png")

# 2. Custom plotting using plot_history
# Shows how to specify metrics explicitly (e.g. diversity)
plot_history(
    pop,
    metrics=["best_fitness", "mean_fitness", "diversity"],
    title="Fitness and Diversity Over Time",
    save_path="figures/02_plotting_custom.png",
    show=True,
)
