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

Note:
    Reproducibility is controlled via the `random_seed` field in the YAML config.
    Set it to an integer for deterministic runs or to null/omit it for stochastic runs.
"""

from evolib import Indiv, Pop, mse_loss, plot_fitness_comparison, rosenbrock


# User-defined fitness function
def my_fitness(indiv: Indiv) -> None:
    expected = [1.0, 1.0, 1.0, 1.0]
    predicted = rosenbrock(indiv.para["test-vector"].vector)
    indiv.fitness = mse_loss(expected, predicted)


def run_experiment(config_path: str) -> Pop:
    pop = Pop(config_path, fitness_function=my_fitness)
    pop.run(verbosity=1)
    return pop


# Run multiple experiments
pop_mutation_constant = run_experiment(config_path="mutation_constant.yaml")
pop_mutation_exponential_decay = run_experiment(config_path="04_exponential_decay.yaml")


# Compare fitness progress
plot_fitness_comparison(
    histories=[pop_mutation_constant, pop_mutation_exponential_decay],
    labels=["Mutation rate static", "Mutation rate decay"],
    metric="best_fitness",
    title="Best Fitness Comparison (constant vs decay)",
    show=True,
    log=True,
    save_path="./figures/04_exponential_decay.png",
)
