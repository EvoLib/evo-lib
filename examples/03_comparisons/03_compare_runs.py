"""
Example 02-03 - Compare Runs

This example demonstrates how to run the same optimization with different settings
(e.g. mutation strength) and compare their results using the fitness history.

Note:
    Reproducibility is controlled via the `random_seed` field in the YAML config.
    Set it to an integer for deterministic runs or to null/omit it for stochastic runs.
"""

from evolib import Indiv, Population, mse_loss, simple_quadratic
from evolib.utils.plotting import plot_fitness_comparison


def my_fitness(indiv: Indiv) -> None:
    expected = 0.0
    predicted = simple_quadratic(indiv.para["test-vector"].vector)
    indiv.fitness = mse_loss(expected, predicted)


def run_experiment(mutation_strength: float) -> Population:
    pop = Population(config_path="03_compare_runs.yaml", fitness_function=my_fitness)

    for indiv in pop.indivs:
        params = indiv.para["test-vector"].evo_params
        params.mutation_strength = mutation_strength

    pop.run(verbosity=0)

    return pop


# Run multiple experiments
pop_low = run_experiment(mutation_strength=0.001)
pop_high = run_experiment(mutation_strength=0.005)

# Compare fitness progress
plot_fitness_comparison(
    histories=[pop_low, pop_high],
    labels=["Mutation σ = 0.001", "Mutation σ = 0.005"],
    metric="best_fitness",
    title="Best Fitness Comparison (Low vs High Mutation)",
    show=True,
    save_path="./figures/03_compare_runs.png",
)
