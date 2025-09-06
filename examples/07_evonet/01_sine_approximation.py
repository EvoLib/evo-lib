"""Approximating sin(x) using a feedforward network defined via EvoNet."""

import numpy as np

from evolib import Indiv, Pop, mse_loss, plot_approximation, save_combined_net_plot

# Define target function
X_RAW = np.linspace(0, 2 * np.pi, 100)
X_NORM = (X_RAW - np.pi) / np.pi
Y_TRUE = np.sin(X_RAW)


# Fitness function
def evonet_fitness(indiv: Indiv) -> None:
    predictions = []
    net = indiv.para["nnet"]

    for x_norm in X_NORM:
        output = net.calc([x_norm])
        predictions.append(output[0])

    indiv.fitness = mse_loss(Y_TRUE, np.array(predictions))


def on_improvement(pop: Pop) -> None:
    indiv = pop.best()
    net = indiv.para["nnet"].net
    y_pred = np.array([net.calc([x])[0] for x in X_NORM])

    save_combined_net_plot(
        net, X_RAW, Y_TRUE, y_pred, f"01_frames/gen{pop.generation_num:04d}.png"
    )


def on_end(pop: Pop) -> None:
    # Visualize result
    y_best = [pop.best().para["nnet"].calc([x])[0] for x in X_NORM]
    plot_approximation(
        y_best, Y_TRUE, title="Best Approximation", pred_marker=None, true_marker=None
    )


# Evolution setup
pop = Pop(
    config_path="configs/01_sine_approximation.yaml", fitness_function=evonet_fitness
)

# Evolution loop
pop.run(on_improvement=on_improvement, on_end=on_end)
