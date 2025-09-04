"""
Example 02-01 - Step-by-Step Evolution (Didactic / Debug Demo)

This script unrolls a single evolutionary generation into explicit steps to
show what high-level strategies (e.g., μ+λ) and `Pop.run()` perform internally.
It is intended for **learning and debugging**, not for production use.

**In real applications, prefer:**
- Pop.run()
- Checkpoint helpers like `resume_or_create(...)` for resumable runs.

What this demo does (one generation, explicitly):
  0) Evaluate parents
  1) Update per-generation parameters (mutation / crossover controls)
  2) Reproduction: clone parents -> offspring
  3) Crossover on offspring
  4) Mutation on offspring
  5) Evaluate offspring
  6) Replacement
  7) Update statistics / logging

The prints after each step are didactic: they expose current vectors and the
effective mutation/crossover settings so you can verify the pipeline behavior.

Reproducibility:
  - set a fixed RNG seed (e.g. np.random.seed(42)).

Verbosity:
  - Adjust `VERBOSITY` to control the amount of printed diagnostics.

Note:
  This file is intentionally more verbose and explicit than recommended for
  normal usage. For real runs and cleaner code, rely on `Pop.run()` or the
  provided strategy helpers and keep step-by-step logic for debugging only.
"""

import numpy as np

from evolib import Indiv, Pop, mse_loss, simple_quadratic
from evolib.operators.crossover import crossover_offspring
from evolib.operators.mutation import mutate_offspring
from evolib.operators.replacement import replace_mu_lambda
from evolib.operators.reproduction import generate_cloned_offspring

np.random.seed(42)


# User-defined fitness function
def my_fitness(indiv: Indiv) -> None:
    """Simple fitness function using the quadratic benchmark and MSE loss."""
    expected = 0.0
    predicted = simple_quadratic(indiv.para["test-vector"].vector)
    indiv.fitness = mse_loss(expected, predicted)


def print_indivs(msg: str, indivs: list[Indiv]) -> None:

    print(msg)
    for i, indiv in enumerate(indivs):
        para = indiv.para["test-vector"]
        evo_params = para.evo_params
        print(
            f"  Indiv {i}: Vector = {para.vector}, "
            f"ms = {evo_params.mutation_strength}, "
            f"mp = {evo_params.mutation_probability}, "
            f"cp = {evo_params.crossover_probability} "
        )


# Create and initialize the population
pop = Pop(config_path="01_step_by_step_evolution.yaml", fitness_function=my_fitness)

# Step 0) Evaluate parents (if needed)
pop.evaluate_fitness()
print_indivs("0) Evaluate parents :", pop.indivs)

# Step 1) Update per-generation parameters (mutation/crossover controls)
pop.update_parameters()
print_indivs("1) Update parameters: ", pop.indivs)

# Step 2) Produce offspring by cloning
offspring = generate_cloned_offspring(pop.indivs, pop.offspring_pool_size)
print_indivs("2) Reproduction (clone parents -> offspring): ", offspring)

# Step 3) Crossover pairs (in-place)
crossover_offspring(pop, offspring)
print_indivs("3) Crossover: ", offspring)

# Step 4) Mutation (in-place)
mutate_offspring(pop, offspring)
print_indivs("4) Mutation: ", offspring)

# Step 5) Evaluate offspring
pop.evaluate_indivs(offspring)
print_indivs("5) Evaluate offspring: ", offspring)

# Step 6) Replacement (μ from parents + offspring)
replace_mu_lambda(pop, pop.indivs + offspring)
print_indivs("6) Replacement: ", pop.indivs)

# Step 7) Stats / logging (increments generation)
pop.update_statistics()

print()
pop.print_status(verbosity=2)
