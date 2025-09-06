"""
Example 04-06 – Adaptive Individual Mutation vs. Static Mutation.

This example compares the effectiveness of adaptive mutation at the individual level
with a static mutation strength. Each individual has its own mutation strength and tau
value that adapts over time.

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


# Run experiments
pop_static = run_experiment("mutation_constant.yaml")
pop_adaptive = run_experiment("06_adaptive_individual.yaml")

# Compare fitness progress
plot_fitness_comparison(
    histories=[pop_static, pop_adaptive],
    labels=["Mutation rate static", "Mutation rate adaptive"],
    metric="best_fitness",
    title="adaptive individual vs. constant",
    show=True,
    log=True,
    save_path="./figures/06_adaptive_individual_vs_static.png",
)
