"""
Example 02-03 — Repeated (mu + lambda)

This example applies the (μ + λ) strategy repeatedly to show how the
population evolves across a few steps.

Didactic only:
    - We call the operator (`evolve_mu_plus_lambda`) directly so you can see
      what happens each step.
    - In real experiments, select the strategy via YAML and call `pop.run()`.
      You do not need to call operators manually.

Requirements:
    - 'population.yaml' present in the current working directory.
"""

from evolib import Indiv, Pop, mse_loss, simple_quadratic
from evolib.operators.strategy import evolve_mu_plus_lambda


def my_fitness(indiv: Indiv) -> None:
    """Assign fitness via a simple quadratic benchmark (distance to zero)."""
    expected = 0.0
    predicted = simple_quadratic(indiv.para["test-vector"].vector)
    indiv.fitness = mse_loss(expected, predicted)


def print_population(pop: Pop, title: str) -> None:
    print(f"\n{title}")
    for i, indiv in enumerate(pop.indivs):
        vector = indiv.para["test-vector"].vector
        print(f"  Indiv {i}: Parameter = {vector}, Fitness = {indiv.fitness:.6f}")


# Initialize population from YAML and attach fitness function
pop = Pop(config_path="population.yaml", fitness_function=my_fitness)

# Evaluate once before the loop
pop.evaluate_fitness()
print_population(pop, "Initial Parents")

# Number of illustrative steps
N_STEPS = 3

for step in range(1, N_STEPS + 1):
    # Apply mu plus lambda
    evolve_mu_plus_lambda(pop)
    print_population(pop, f"After Mu Plus Lambda - Step {step}")
