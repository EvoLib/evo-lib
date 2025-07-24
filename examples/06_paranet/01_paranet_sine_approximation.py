# 01_netvector_demo.py
"""
Minimal working example: Using NetVector in an evolutionary regression task.
Approximates the function f(x) = sin(x) over [0, 2π] using a simple neural network.
"""

import matplotlib.pyplot as plt
import numpy as np

from evolib import Indiv, Pop, mse_loss
from evolib.representation.netvector import NetVector

# Target data: y = sin(x) over [0, 2π]
X_RANGE = np.linspace(0, 2 * np.pi, 100)
Y_TRUE = np.sin(X_RANGE)

# Create NetVector model (shared across individuals)
netvector = NetVector(input_dim=1, hidden_dim=5, output_dim=1, activation="tanh")


def netvector_fitness(indiv: Indiv) -> None:
    """
    Fitness function using NetVector to approximate sin(x). Applies MSE loss over
    sampled input points.

    Args:
        indiv (Indiv): The individual to evaluate.
    """
    predictions: list[float] = []

    for x in X_RANGE:
        x_input = np.array([x])
        y_pred = netvector.forward(x_input, indiv.para["nnet"].vector)
        predictions.append(y_pred.item())

    y_pred_array = np.array(predictions)
    indiv.fitness = mse_loss(Y_TRUE, y_pred_array)


# Evolutionary run
pop = Pop(config_path="configs/01_netvector_sine_approximation.yaml")
pop.set_functions(fitness_function=netvector_fitness)

for _ in range(pop.max_generations):
    pop.run_one_generation()
    pop.print_status()


# Visualize best solution
best = pop.best()
y_best = [
    netvector.forward(np.array([x]), best.para["nnet"].vector).item() for x in X_RANGE
]

plt.plot(X_RANGE, Y_TRUE, label="Target: sin(x)")
plt.plot(X_RANGE, y_best, label="Best Approximation", linestyle="--")
plt.title("NetVector Fit to sin(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
