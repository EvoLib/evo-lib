"""
Example 03 – VectorNet with gain and bias modulation.

The controller evolves gain and bias while VectorNet evolves the fixed network
parameters. The output is ``gain * net(x) + bias``.
"""

import numpy as np

from evolib import Indiv, Pop, mse_loss, plot_approximation

CONFIG = "configs/03_netvector_gain_and_bias.yaml"

X_RANGE = np.linspace(0, 2 * np.pi, 100)
Y_TARGET = 0.8 * np.sin(X_RANGE) + 0.2


def fitness_gain_bias(indiv: Indiv) -> None:
    controller = indiv.para["controller"]
    nnet = indiv.para["nnet"]

    gain = controller.vector[0]
    bias = controller.vector[1]

    predictions = []
    for x in X_RANGE:
        y = nnet.forward(np.array([x]))
        predictions.append((gain * y + bias).item())

    indiv.fitness = mse_loss(Y_TARGET, np.array(predictions))


def show_approximation_plot(pop: Pop) -> None:
    best = pop.best()
    controller = best.para["controller"]
    nnet = best.para["nnet"]

    gain = controller.vector[0]
    bias = controller.vector[1]
    y_pred = [gain * nnet.forward(np.array([x])).item() + bias for x in X_RANGE]

    plot_approximation(
        y_pred,
        Y_TARGET,
        title="VectorNet with Gain + Bias Modulation (Target: 0.8·sin(x)+0.2)",
        pred_label="Approximation",
        show=True,
        show_grid=False,
        x_vals=X_RANGE,
    )


pop = Pop(CONFIG, fitness_function=fitness_gain_bias)
pop.run(on_end=show_approximation_plot)
