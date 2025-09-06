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

Note:
    Reproducibility is controlled via the `random_seed` field in the YAML config.
    Set it to an integer for deterministic runs or to null/omit it for stochastic runs.
"""

from evolib import Indiv, Pop, mse_loss, plot_fitness_comparison, rosenbrock


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
pop_mutation_adaptive = run_experiment(config_path="05_adaptive_global.yaml")


# Compare fitness progress
plot_fitness_comparison(
    histories=[pop_mutation_constant, pop_mutation_adaptive],
    labels=["Mutation rate static", "Mutation rate adaptive"],
    metric="best_fitness",
    title="Best Fitness Comparison (constant vs adaptive)",
    show=True,
    log=True,
    save_path="./figures/05_adaptive_global.png",
)
