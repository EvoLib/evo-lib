"""
Example 03 – netvector + Gain and Bias Modulation.

This example evolves a composite individual with:
- a 'controller' vector of 2 scalars: gain and bias
- a 'nnet' NetVector

The network output is scaled and shifted by the controller:
    y_hat = gain * net(x) + bias

Target: f(x) = 0.8 * sin(x) + 0.2
"""

import matplotlib.pyplot as plt
import numpy as np

from evolib import Indiv, Pop, mse_loss
from evolib.representation.netvector import NetVector

# Target function
X_RANGE = np.linspace(0, 2 * np.pi, 100)
Y_TARGET = 0.8 * np.sin(X_RANGE) + 0.2

# Shared network structure
netvector = NetVector(input_dim=1, hidden_dim=5, output_dim=1, activation="tanh")


def fitness_gain_bias(indiv: Indiv) -> None:
    gain = indiv.para["controller"].vector[0]
    bias = indiv.para["controller"].vector[1]
    net_vector = indiv.para["nnet"].vector

    y_preds = []
    for x in X_RANGE:
        x_input = np.array([x])
        y = netvector.forward(x_input, net_vector)
        y_modulated = gain * y + bias
        y_preds.append(y_modulated.item())

    y_pred_array = np.array(y_preds)
    indiv.fitness = mse_loss(Y_TARGET, y_pred_array)


def run(config_path: str) -> Pop:
    pop = Pop(config_path)
    pop.set_functions(fitness_function=fitness_gain_bias)

    for _ in range(pop.max_generations):
        pop.run_one_generation()
        pop.print_status()

    return pop


if __name__ == "__main__":
    pop = run("configs/03_netvector_gain_and_bias.yaml")
    best = pop.best()

    gain = best.para["controller"].vector[0]
    bias = best.para["controller"].vector[1]
    net_vector = best.para["nnet"].vector

    y_pred = [
        gain * netvector.forward(np.array([x]), net_vector).item() + bias
        for x in X_RANGE
    ]

    plt.plot(X_RANGE, Y_TARGET, label="Target: 0.8·sin(x)+0.2")
    plt.plot(X_RANGE, y_pred, "--", label=f"Best (gain={gain:.2f}, bias={bias:.2f})")
    plt.title("NetVector with Gain + Bias Modulation")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
