"""
Example 3.0 - History

This example shows how to print per-generation statistics as a console table
via `pop.print_history()`. A Pandas DataFrame is available via `pop.history_df`.
"""

from evolib import Indiv, Pop, mse_loss, simple_quadratic


def my_fitness(indiv: Indiv) -> None:
    """Simple fitness function using the quadratic benchmark and MSE loss."""
    expected = 0.0
    predicted = simple_quadratic(indiv.para["test-vector"].vector)
    indiv.fitness = mse_loss(expected, predicted)


# Load configuration and initialize population
pop = Pop(config_path="population.yaml", fitness_function=my_fitness)

# Run full evolution using configured strategy
pop.run(verbosity=0)

# Print history
pop.print_history()

print("\nFinal History Snapshot (last 5 generations):")
pop.print_history(last_n=5)

# Optional: export for plotting
# pop.history_logger.save_csv("history.csv")
