"""
NetVector usage: Approximating sin(x) using a feedforward network defined via
Vector.  The network structure is configured in YAML using dim_type = 'net'
and interpreted with NetVector at evaluation time.
"""

import numpy as np

from evolib import Indiv, Pop, mse_loss, plot_approximation
from evolib.representation.netvector import NetVector

# Define target function
X_RANGE = np.linspace(0, 2 * np.pi, 100)
Y_TRUE = np.sin(X_RANGE)


# Fitness function using NetVector to interpret Vector
def netvector_fitness(indiv: Indiv) -> None:
    predictions: list[float] = []

    for x in X_RANGE:
        x_input = np.array([x])
        y_pred = net.forward(x_input, indiv.para["nnet"].vector)
        predictions.append(y_pred.item())

    indiv.fitness = mse_loss(Y_TRUE, np.array(predictions))


def on_end(pop: Pop) -> None:
    # Final visualization

    best = pop.best()
    y_best = [
        net.forward(np.array([x]), best.para["nnet"].vector).item() for x in X_RANGE
    ]
    plot_approximation(
        y_best, Y_TRUE, title="Best Approximation", pred_marker=None, true_marker=None
    )


# Run evolution
pop = Pop(
    config_path="configs/01_netvector_sine_approximation.yaml",
    fitness_function=netvector_fitness,
)

net = NetVector.from_config(pop.config, module="nnet")

pop.run(on_end=on_end)
