"""
Example 02-05– Adaptive Global Mutation.

This example demonstrates the use of an adaptive global mutation strategy within
a (mu + lmbda) evolutionary algorithm framework. The mutation strength is updated
globally based on the population configuration, allowing the mutation process to adapt
over time.

Key Elements:
- The benchmark problem is based on the 4-dimensional Rosenbrock function.
- Fitness is computed as the mean squared error between predicted and expected values.
- Gaussian mutation is applied to each individual’s parameters.
"""

from evolib import (
    Indiv,
    MutationParams,
    Pop,
    Strategy,
    evolve_mu_lambda,
    mse_loss,
    mutate_gauss,
    rosenbrock,
)


# User-defined fitness function
def my_fitness(indiv: Indiv) -> None:
    expected = [1.0, 1.0, 1.0, 1.0]
    predicted = rosenbrock(indiv.para.vector)
    indiv.fitness = mse_loss(expected, predicted)


def run_experiment(config_path: str) -> None:
    pop = Pop(config_path)
    pop.initialize_population()
    pop.set_functions(fitness_function=my_fitness)

    for _ in range(pop.max_generations):
        evolve_mu_lambda(pop, strategy=Strategy.MU_PLUS_LAMBDA)

        pop.print_status(verbosity=1)
        print(f"   DiversityEMA: {pop.diversity_ema:.4f}  | "
              f"MinDiversityThreshold: {pop.indivs[0].para.min_diversity_threshold} | "
              f"MaxDiversityThreshold: {pop.indivs[0].para.max_diversity_threshold}")
        print(f"   MutationStrength: {pop.indivs[0].para.mutation_strength:.4f}")


# Run multiple experiments
print("With static mutation strength:\n")
run_experiment(config_path="04_rate_constant.yaml")

print("\n\nWith adaptive mutation strength:\n")
run_experiment(config_path="05_adaptive_global.yaml")
