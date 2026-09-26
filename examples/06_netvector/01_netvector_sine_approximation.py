"""Approximate sin(x) using a fixed-topology VectorNet."""

import numpy as np

from evolib import Indiv, Pop, mse_loss, plot_approximation

CONFIG_FILE = "configs/01_netvector_sine_approximation.yaml"

X_RANGE = np.linspace(0, 2 * np.pi, 100)
Y_TRUE = np.sin(X_RANGE)


def vectornet_fitness(indiv: Indiv) -> None:
    nnet = indiv.para["nnet"]

    predictions: list[float] = []
    for x in X_RANGE:
        y_pred = nnet.forward(np.array([x]))
        predictions.append(y_pred.item())

    indiv.fitness = mse_loss(Y_TRUE, np.array(predictions))


def on_end(pop: Pop) -> None:
    best = pop.best()
    nnet = best.para["nnet"]

    y_best = [nnet.forward(np.array([x])).item() for x in X_RANGE]
    plot_approximation(
        y_best,
        Y_TRUE,
        title="Best Approximation",
        pred_marker=None,
        true_marker=None,
    )


pop = Pop(CONFIG_FILE, fitness_function=vectornet_fitness)
pop.run(on_end=on_end)
