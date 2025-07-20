"""
Example 04-06 – Adaptive Individual Mutation vs. Static Mutation.

This example compares the effectiveness of adaptive mutation at the individual level
with a static mutation strength. Each individual has its own mutation strength and tau
value that adapts over time.
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

    return pop.history_logger.to_dataframe()


# Run experiments
history_static = run_experiment("mutation_constant.yaml")
history_adaptive = run_experiment("06_adaptive_individual.yaml")

# Compare fitness progress
plot_fitness_comparison(
    histories=[history_static, history_adaptive],
    labels=["Mutation rate static", "Mutation rate adaptive"],
    metric="best_fitness",
    title="adaptive individual vs. constant",
    show=True,
    log=True,
    save_path="./figures/06_adaptive_individual_vs_static.png",
)
