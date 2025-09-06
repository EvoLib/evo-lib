"""
Demonstrates structural mutation on the classic XOR problem using EvoLib's EvoNet.

This example evolves neural networks that approximate the XOR function. Structural
mutation operators (add/remove neuron, add/remove connection, etc.) allow the network
to gradually grow and adapt its topology instead of working with a fixed architecture.

Workflow:
    1. Define the XOR dataset.
    2. Implement a fitness function (mean squared error on XOR targets).
    3. Initialize a population from a YAML configuration.
    4. Run evolution for a number of generations.
    5. Save intermediate network visualizations whenever a new best fitness is found.
    6. Plot the final network output vs. the target XOR values.

Expected outcome:
    - Over generations, the network topology mutates and adapts.
    - The best individual should approximate XOR with low error.
"""

import numpy as np

from evolib import (
    Indiv,
    Pop,
    mse_loss,
    plot_approximation,
    resume_or_create,
    save_checkpoint,
    save_combined_net_plot,
)

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([0, 1, 1, 0])

# Normalization
X_NORM = X.astype(float)
Y_TRUE = Y.astype(float)


def xor_fitness(indiv: Indiv) -> None:
    """
    Fitness function for the XOR task.

    Computes the mean squared error (MSE) between network predictions
    and the true XOR outputs. Lower values indicate better performance.

    Args:
        indiv (Indiv): An individual containing a 'brain' EvoNet module.
    """
    net = indiv.para["brain"]
    predictions = [net.calc(x.tolist())[0] for x in X_NORM]
    indiv.fitness = mse_loss(Y_TRUE, np.array(predictions))


def on_improvement(pop: Pop) -> None:
    indiv = pop.best()
    net = indiv.para["brain"].net
    y_pred = [net.calc(x.tolist())[0] for x in X]
    save_combined_net_plot(
        net,
        np.arange(len(X)),
        Y_TRUE,
        np.array(y_pred),
        f"03_frames/gen_{pop.generation_num:04d}.png",
        title="Structural Mutation on XOR",
    )

    save_checkpoint(pop, run_name="xor")


def on_end(pop: Pop) -> None:
    # Final visualization
    best = pop.best()
    net = best.para["brain"].net
    y_pred = [net.calc(x.tolist())[0] for x in X_NORM]
    plot_approximation(y_pred, Y_TRUE, title="Best XOR Approximation")


# Evolution setup
pop = resume_or_create("configs/03_structural_xor.yaml", xor_fitness, "xor")

# Evolution loop
pop.run(on_improvement=on_improvement, on_end=on_end)
