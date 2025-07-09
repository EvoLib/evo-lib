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


Configurations Compared:
04_rate_constant.yaml:
    Mutation strategy: CONSTANT
    Strength & rate remain unchanged across generations

05_adaptive_global.yaml:
    Mutation strategy: ADAPTIVE_GLOBAL
    Strength & rate are adjusted based on diversity (EMA)
"""

from evolib import Indiv, Pop, evolve_mu_plus_lambda, mse_loss, rosenbrock


# User-defined fitness function
def my_fitness(indiv: Indiv) -> None:
    expected = [1.0, 1.0, 1.0, 1.0]
    predicted = rosenbrock(indiv.para.vector)
    indiv.fitness = mse_loss(expected, predicted)


def run_experiment(config_path: str) -> None:
    pop = Pop(config_path)
    pop.set_functions(fitness_function=my_fitness)

    for _ in range(pop.max_generations):
        evolve_mu_plus_lambda(pop)

        pop.print_status(verbosity=1)
        best = pop.best()
        print(
            f"   DiversityEMA: {pop.diversity_ema:.4f}  | "
            f"MinDiversityThreshold: {best.para.min_diversity_threshold} | "
            f"MaxDiversityThreshold: {best.para.max_diversity_threshold}"
        )
        print(f"   MutationStrength: {best.para.mutation_strength:.4f}")


# Configuration with constant mutation strength
print("Case A: Constant mutation strength\n")
run_experiment(config_path="05_rate_constant.yaml")

# Use an adaptive global mutation strategy (modifies strength & probability
# depending on diversity)
print("\n\nCase B: Adaptive global mutation strength:\n")
run_experiment(config_path="05_adaptive_global.yaml")
