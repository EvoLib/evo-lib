"""
Example 01-04 — Defining a Fitness Function.

This example demonstrates:
- How to provide a custom fitness function
- How EvoLib uses it automatically during evolution
"""

import numpy as np

from evolib import Indiv, resume_or_create


# Sphere fitness: minimize sum of squares
def sphere_fitness(indiv: Indiv) -> None:
    vec = np.array(indiv.para["test-vector"].vector, dtype=float)
    indiv.fitness = float(np.sum(vec * vec))


def main() -> None:
    # Either direct:
    # pop = Population(config_path="population.yaml", fitness_function=sphere_fitness)

    # Or resumable (preferred for real runs):
    pop = resume_or_create("population.yaml", sphere_fitness, run_name="sphere_demo")
    pop.run()


if __name__ == "__main__":
    main()
