"""
Example 02-06 – Adaptive Individual Mutation with Tau.

This example demonstrates the use of an adaptive individual mutation strategy where
each individual maintains its own mutation parameters (e.g., mutation strength) and
adapts them independently over time using a log-normal update (self-adaptation).

Key Elements:
- The benchmark problem is based on the 4-dimensional Rosenbrock function.
- Each individual maintains its own mutation strength and learning rate.
- Log-normal updates are used for self-adaptation.

Configuration:
06_adaptive_individual.yaml:
    Mutation strategy: ADAPTIVE_INDIVIDUAL
    Each individual carries and evolves its own mutation strength
"""

from evolib import Indiv, Pop, mse_loss, rosenbrock
from evolib.operators.strategy import evolve_mu_plus_lambda


# User-defined fitness function
def my_fitness(indiv: Indiv) -> None:
    expected = [1.0, 1.0, 1.0, 1.0]
    predicted = rosenbrock(indiv.para["test-vector"].vector)
    indiv.fitness = mse_loss(expected, predicted)


def run_experiment(config_path: str) -> None:
    pop = Pop(config_path)
    pop.set_functions(fitness_function=my_fitness)

    for _ in range(pop.max_generations):
        evolve_mu_plus_lambda(pop)

        pop.print_status(verbosity=1)
        best_params = pop.best().para["test-vector"].evo_params
        print(f"   tau: {best_params.tau:.4f}")
        print(f"   MutationStrength: {best_params.mutation_strength:.4f}")


print("Running adaptive_individual experiment with tau...\n")
run_experiment("06_adaptive_individual.yaml")
