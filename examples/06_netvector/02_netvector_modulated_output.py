"""
Example 02 – ParaComposite with a Vector controller and VectorNet.

The controller evolves one scalar gain. VectorNet evolves the fixed network parameters.
The output is ``gain * net(x)``.
"""

import numpy as np

from evolib import Indiv, Pop, mse_loss, plot_approximation

CONFIG = "configs/02_netvector_modulated_output.yaml"

X_RANGE = np.linspace(0, 2 * np.pi, 100)
Y_TRUE = np.sin(X_RANGE)


def composite_fitness(indiv: Indiv) -> None:
    controller = indiv.para["controller"]
    nnet = indiv.para["nnet"]

    gain = controller.vector[0]
    y_preds = []
    for x in X_RANGE:
        y = nnet.forward(np.array([x]))
        y_preds.append((gain * y).item())

    indiv.fitness = mse_loss(Y_TRUE, np.array(y_preds))


def show_approximation_plot(pop: Pop) -> None:
    best = pop.best()
    controller = best.para["controller"]
    nnet = best.para["nnet"]

    gain = controller.vector[0]
    y_pred = [gain * nnet.forward(np.array([x])).item() for x in X_RANGE]

    plot_approximation(
        y_pred,
        Y_TRUE,
        title="Function approximation - Modulated VectorNet Output",
        pred_label="Approximation",
        show=True,
        show_grid=False,
        x_vals=X_RANGE,
    )


pop = Pop(CONFIG, fitness_function=composite_fitness)
pop.run(on_end=show_approximation_plot)
