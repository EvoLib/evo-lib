"""
Example 03 – NetVector + Gain and Bias Modulation.

This example evolves a composite individual with:
- a 'controller' Vector of 2 scalars: gain and bias
- a 'nnet' Vector interpreted as feedforward network

The output is scaled and shifted:
    ŷ = gain * net(x) + bias

Target: f(x) = 0.8 * sin(x) + 0.2
"""

import matplotlib.pyplot as plt
import numpy as np

from evolib import Indiv, Pop, mse_loss
from evolib.representation.netvector import NetVector

# Target function: sin(x) scaled and shifted
X_RANGE = np.linspace(0, 2 * np.pi, 100)
Y_TARGET = 0.8 * np.sin(X_RANGE) + 0.2


def fitness_gain_bias(indiv: Indiv) -> None:
    gain = indiv.para["controller"].vector[0]
    bias = indiv.para["controller"].vector[1]
    net_vector = indiv.para["nnet"].vector

    predictions = []
    for x in X_RANGE:
        x_input = np.array([x])
        y = net.forward(x_input, net_vector)
        y_modulated = gain * y + bias
        predictions.append(y_modulated.item())

    indiv.fitness = mse_loss(Y_TARGET, np.array(predictions))


# Run evolution
pop = Pop("configs/03_netvector_gain_and_bias.yaml")
pop.set_functions(fitness_function=fitness_gain_bias)

net = NetVector.from_config(pop.config, module="nnet")

for _ in range(pop.max_generations):
    pop.run_one_generation()
    pop.print_status()

# Visualize result
best = pop.best()

gain = best.para["controller"].vector[0]
bias = best.para["controller"].vector[1]
net_vector = best.para["nnet"].vector

y_pred = [gain * net.forward(np.array([x]), net_vector).item() + bias for x in X_RANGE]

plt.plot(X_RANGE, Y_TARGET, label="Target: 0.8·sin(x)+0.2")
plt.plot(X_RANGE, y_pred, "--", label=f"Best (gain={gain:.2f}, bias={bias:.2f})")
plt.title("NetVector with Gain + Bias Modulation")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
