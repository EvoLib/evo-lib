"""
Example 04-02 – Adaptive Global Mutation.

This example demonstrates the use of an adaptive global mutation strategy within
a (mu + lmbda) evolutionary algorithm framework. The mutation strength is updated
globally based on the population configuration, allowing the mutation process to adapt
over time.

Key Elements:
- The benchmark problem is based on the 4-dimensional Rosenbrock function.
- Fitness is computed as the mean squared error between predicted and expected values.
- Gaussian mutation is applied to each individual’s parameters.
- The experiment compares static vs. adaptive mutation rate strategies.
- Results are visualized using a fitness comparison plot across generations.

- Results are saved to './figures/05_adaptive_global.png'
"""

import pandas as pd

from evolib import Indiv, Pop, mse_loss, plot_fitness_comparison, rosenbrock


def my_fitness(indiv: Indiv) -> None:
    expected = [1.0, 1.0, 1.0, 1.0]
    predicted = rosenbrock(indiv.para.vector)
    indiv.fitness = mse_loss(expected, predicted)


def run_experiment(config_path: str) -> pd.DataFrame:
    pop = Pop(config_path)
    pop.set_functions(fitness_function=my_fitness)

    for _ in range(pop.max_generations):
        pop.run_one_generation()

    pd.set_option("display.max_rows", None)
    history = pop.history_logger.to_dataframe()
    print(history)

    print(f"Best Indiduum Parameter: {pop.indivs[0].para}")

    return pop.history_logger.to_dataframe()


# Run multiple experiments
history_mutation_constant = run_experiment(config_path="mutation_constant.yaml")
history_mutation_adaptive = run_experiment(config_path="05_adaptive_global.yaml")


# Compare fitness progress
plot_fitness_comparison(
    histories=[history_mutation_constant, history_mutation_adaptive],
    labels=["Mutation rate static", "Mutation rate adaptive"],
    metric="best_fitness",
    title="Best Fitness Comparison (constant vs adaptive)",
    show=True,
    log=True,
    save_path="./figures/05_adaptive_global.png",
)
