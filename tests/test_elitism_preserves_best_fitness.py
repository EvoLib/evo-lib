import random

import numpy as np

from evolib import Indiv, Pop, mse_loss, rastrigin


def fitness_function(indiv: Indiv) -> None:
    target = np.zeros(indiv.para["vector"].dim)
    predicted = rastrigin(indiv.para["vector"].vector)
    indiv.fitness = mse_loss(target, predicted)


def test_elitism_preserves_best_fitness() -> None:
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    pop = Pop(config_path="./tests/configs/elitism_test.yaml")
    pop.set_functions(fitness_function=fitness_function)

    best_fitnesses = []
    for _ in range(pop.max_generations):
        pop.run_one_generation()
        best_fitnesses.append(pop.best_fitness)

    # Check that best_fitness is monotonically non-increasing
    for i in range(1, len(best_fitnesses)):
        assert best_fitnesses[i] <= best_fitnesses[i - 1], (
            f"Elitism violated: best_fitness increased from "
            f"{best_fitnesses[i - 1]:.4f} to {best_fitnesses[i]:.4f} "
            f"at generation {i}"
        )
