"""
Example 02-04 – Exponential Decay of Mutation Rate.

This example demonstrates the impact of exponentially decaying mutation rates on the
performance of a (μ + λ) evolution strategy. It compares a static mutation rate with an
exponentially decreasing one using the Rosenbrock function as the fitness landscape.

The script runs two experiments with different population configurations and prints the
resulting fitness progression over generations.
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
        print(
            f"   MutationStrength: "
            f"{pop.indivs[0].para['test-vector'].evo_params.mutation_strength:.4f}"
        )


# Run multiple experiments
print("With static mutation strength:\n")
run_experiment(config_path="04_rate_constant.yaml")

print("\n\nWith exponential decay mutation strength:\n")
run_experiment(config_path="04_exponential_decay.yaml")
