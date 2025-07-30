"""
NetVector usage: Approximating sin(x) using a feedforward network defined via
ParaVector.  The network structure is configured in YAML using dim_type = 'net'
and interpreted with NetVector at evaluation time.
"""

import matplotlib.pyplot as plt
import numpy as np

from evolib import Indiv, Pop, mse_loss
from evolib.representation.netvector import NetVector

# Define target function
X_RANGE = np.linspace(0, 2 * np.pi, 100)
Y_TRUE = np.sin(X_RANGE)


# Fitness function using NetVector to interpret ParaVector
def netvector_fitness(indiv: Indiv) -> None:
    predictions: list[float] = []

    for x in X_RANGE:
        x_input = np.array([x])
        y_pred = net.forward(x_input, indiv.para["nnet"].vector)
        predictions.append(y_pred.item())

    indiv.fitness = mse_loss(Y_TRUE, np.array(predictions))


# Run evolution
pop = Pop(config_path="configs/01_netvector_sine_approximation.yaml")
pop.set_functions(fitness_function=netvector_fitness)

net = NetVector.from_config(pop.config, module="nnet")

for _ in range(pop.max_generations):
    pop.run_one_generation()
    pop.print_status()


# Visualize result
best = pop.best()
y_best = [net.forward(np.array([x]), best.para["nnet"].vector).item() for x in X_RANGE]

plt.plot(X_RANGE, Y_TRUE, label="Target: sin(x)")
plt.plot(X_RANGE, y_best, label="Best Approximation", linestyle="--")
plt.title("NetVector Fit to sin(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
