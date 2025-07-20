"""
Example 3.0 - History

This example demonstrates how to log and inspect fitness statistics during evolution.

It introduces:
- Logging of per-generation statistics (e.g., best, mean, std)
- Accessing and printing the history as a DataFrame

Note:
  In earlier examples, we explicitly called strategy functions like
  `evolve_mu_plus_lambda(pop)` to make the underlying logic transparent.
  Starting from this point, we use the encapsulated `pop.run_one_generation()` method
  to reflect a more modular and extensible design, where the strategy is defined via
  configuration.

Requirements:
    'population.yaml' must be present in the current working directory
"""

from evolib import Indiv, Pop, mse_loss, simple_quadratic


def my_fitness(indiv: Indiv) -> None:
    """Simple fitness function using the quadratic benchmark and MSE loss."""
    expected = 0.0
    predicted = simple_quadratic(indiv.para.vector)
    indiv.fitness = mse_loss(expected, predicted)


def print_population(pop: Pop, title: str) -> None:
    print(f"{title}")
    for i, indiv in enumerate(pop.indivs):
        print(
            f"  Indiv {i}: Parameter = {indiv.para.vector},"
            f"Fitness = {indiv.fitness:.6f}"
        )


# Load configuration and initialize population
pop = Pop(config_path="population.yaml")
pop.set_functions(fitness_function=my_fitness)

print_population(pop, "Initial Parents")

# Mu Plus Lambda
for gen in range(pop.max_generations):
    pop.run_one_generation()

history = pop.history_logger.to_dataframe()
print(
    history[
        [
            "generation",
            "best_fitness",
            "worst_fitness",
            "mean_fitness",
            "std_fitness",
            "iqr_fitness",
        ]
    ]
)

print("\nFinal History Snapshot (last 5 generations):")
print(
    history[
        [
            "generation",
            "best_fitness",
            "worst_fitness",
            "mean_fitness",
            "std_fitness",
            "iqr_fitness",
        ]
    ].tail()
)

best_overall = history["best_fitness"].min()
print(f"\nBest fitness achieved: {best_overall:.6f}")
