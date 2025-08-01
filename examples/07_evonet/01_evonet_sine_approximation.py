"""Approximating sin(x) using a feedforward network defined via EvoNet."""

import matplotlib.pyplot as plt
import numpy as np

from evolib import Indiv, Pop, mse_loss

# Define target function
X_RANGE = np.linspace(0, 2 * np.pi, 100)
Y_TRUE = np.sin(X_RANGE)


# Fitness function
def evonet_fitness(indiv: Indiv) -> None:
    predictions: list[float] = []

    for x in X_RANGE:
        x_input = np.array([x])
        output = indiv.para["nnet"].calc(x_input)
        y_pred = output[0]
        predictions.append(y_pred)

    indiv.fitness = mse_loss(Y_TRUE, np.array(predictions))


# Run evolution
pop = Pop(config_path="configs/01_evonet_sine_approximation.yaml")
pop.set_functions(fitness_function=evonet_fitness)


for _ in range(pop.max_generations):
    pop.run_one_generation()
    pop.print_status()

# Visualize result
best = pop.best()
y_best = [best.para["nnet"].calc(np.array([x])) for x in X_RANGE]

plt.plot(X_RANGE, Y_TRUE, label="Target: sin(x)")
plt.plot(X_RANGE, y_best, label="Best Approximation", linestyle="--")
plt.title("EvoNet Fit to sin(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
