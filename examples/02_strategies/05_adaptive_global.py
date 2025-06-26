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
    predicted = rosenbrock(indiv.para)
    indiv.fitness = mse_loss(expected, predicted)


# User-defined mutation function
def my_mutation(indiv: Indiv, params: MutationParams) -> None:
    """
    Mutates the individual's parameters using Gaussian mutation.

    Args:
        indiv (Indiv): The individual to mutate.
        params (MutationParams): Mutation context object that includes:
            - strength (float): Standard deviation of the Gaussian mutation.
            - bounds (tuple[float, float]): Value limits for the mutation.
            - rate (Optional[float]): (Unused) mutation rate, if applicable.
            - bias (Optional[float]): (Unused) bias mutation strength for
              neural networks.
    """
    for idx, _ in enumerate(indiv.para):
        indiv.para[idx] = mutate_gauss(indiv.para[idx], params.strength, params.bounds)


def initialize_population(pop: Pop) -> None:
    for _ in range(pop.parent_pool_size):
        new_indiv = pop.create_indiv()
        new_indiv.para = [-2.0, 2.0, -1.5, 1.5]  # for better comparison reasons
        pop.add_indiv(new_indiv)
    for indiv in pop.indivs:
        my_fitness(indiv)


def run_experiment(config_path: str) -> None:
    my_pop = Pop(config_path)
    initialize_population(my_pop)

    for _ in range(my_pop.max_generations):
        evolve_mu_lambda(
            my_pop, my_fitness, my_mutation, strategy=Strategy.MU_PLUS_LAMBDA
        )

        my_pop.print_status(verbosity=3)
        print()


# Run multiple experiments
print("With static mutation strength:\n")
run_experiment(config_path="population.yaml")

print("\n\nWith adaptive mutation strength:\n")
run_experiment(config_path="05_adaptive_global.yaml")
