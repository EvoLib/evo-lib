"""
Example 02 – ParaComposite with controller + NetVector.

This example evolves a composite individual with:
- a 'controller' vector (1D), used to modulate network output (gain)
- a 'nnet' Vector interpreted as feedforward NetVector

The fitness is the MSE between gain * net(x) and sin(x).
"""

import numpy as np

from evolib import Indiv, Pop, mse_loss, plot_approximation
from evolib.representation.netvector import NetVector

CONFIG = "configs/02_netvector_modulated_output.yaml"

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


def show_approximation_plot(pop: Pop) -> None:
    # Visualize result
    best = pop.best()

    gain = best.para["controller"].vector[0]
    net_vector = best.para["nnet"].vector
    y_pred = [gain * net.forward(np.array([x]), net_vector).item() for x in X_RANGE]

    plot_approximation(
        y_pred,
        Y_TRUE,
        title="Function approximation - Modulated NetVector Output",
        pred_label="Approximation",
        show=True,
        show_grid=False,
        x_vals=X_RANGE,
    )


# Run evolution
pop = Pop(CONFIG, fitness_function=composite_fitness)
net = NetVector.from_config(pop.config, module="nnet")
pop.run(on_end=show_approximation_plot)
