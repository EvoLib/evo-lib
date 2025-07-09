"""
Example 04-01 – Exponential Decay of Mutation Rate.

This example demonstrates the impact of exponentially decaying mutation rates
on the performance of a (μ + λ) evolution strategy. It compares a static mutation
rate with an exponentially decreasing one using the Rosenbrock function as the fitness
landscape.

The script runs two experiments with different population configurations and visualizes
the resulting fitness progression over generations.

Visualization:
- A comparison plot of best fitness per generation is saved under:
'./figures/04_exponential_decay.png'
"""

import pandas as pd

from evolib import Indiv, Pop, mse_loss, plot_fitness_comparison, rosenbrock


# User-defined fitness function
def my_fitness(indiv: Indiv) -> None:
    expected = [1.0, 1.0, 1.0, 1.0]
    predicted = rosenbrock(indiv.para.vector)
    indiv.fitness = mse_loss(expected, predicted)


def run_experiment(config_path: str) -> pd.DataFrame:
    pop = Pop(config_path)
    pop.set_functions(fitness_function=my_fitness)

    for _ in range(pop.max_generations):
        pop.run_one_generation()

    history = pop.history_logger.to_dataframe()
    print(history)

    print(f"Best Indiduum Parameter: {pop.indivs[0].para}")

    return pop.history_logger.to_dataframe()


# Run multiple experiments
history_mutation_constant = run_experiment(config_path="mutation_constant.yaml")
history_mutation_exponential_decay = run_experiment(
    config_path="04_exponential_decay.yaml"
)


# Compare fitness progress
plot_fitness_comparison(
    histories=[history_mutation_constant, history_mutation_exponential_decay],
    labels=["Mutation rate static", "Mutation rate decay"],
    metric="best_fitness",
    title="Best Fitness Comparison (constant vs decay)",
    show=True,
    log=True,
    save_path="./figures/04_exponential_decay.png",
)
