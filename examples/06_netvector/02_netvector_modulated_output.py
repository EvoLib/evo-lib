"""
Example 02 – ParaComposite with controller + NetVector.

This example evolves a composite individual with:
- a 'controller' vector (1D), used to modulate network output (gain)
- a 'nnet' Vector interpreted as feedforward NetVector

The fitness is the MSE between gain * net(x) and sin(x).
"""

import matplotlib.pyplot as plt
import numpy as np

from evolib import Indiv, Pop, mse_loss
from evolib.representation.netvector import NetVector

# Target function
X_RANGE = np.linspace(0, 2 * np.pi, 100)
Y_TRUE = np.sin(X_RANGE)


# Fitness function
def composite_fitness(indiv: Indiv) -> None:
    gain = indiv.para["controller"].vector[0]
    net_vector = indiv.para["nnet"].vector

    y_preds = []
    for x in X_RANGE:
        x_input = np.array([x])
        y = net.forward(x_input, net_vector)
        y_modulated = gain * y
        y_preds.append(y_modulated.item())

    indiv.fitness = mse_loss(Y_TRUE, np.array(y_preds))


# Run evolution
pop = Pop("configs/02_netvector_modulated_output.yaml")
pop.set_functions(fitness_function=composite_fitness)

net = NetVector.from_config(pop.config, module="nnet")

for _ in range(pop.max_generations):
    pop.run_one_generation()
    pop.print_status()


# Visualize result
best = pop.best()

gain = best.para["controller"].vector[0]
net_vector = best.para["nnet"].vector

y_final = [gain * net.forward(np.array([x]), net_vector).item() for x in X_RANGE]

plt.plot(X_RANGE, Y_TRUE, label="Target: sin(x)")
plt.plot(X_RANGE, y_final, "--", label=f"Best (gain={gain:.2f})")
plt.title("Modulated NetVector Output")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
