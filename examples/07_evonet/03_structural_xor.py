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

from evolib import Indiv, Pop, mse_loss, plot_approximation, save_combined_net_plot

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


# Evolution setup
pop = Pop(config_path="configs/03_structural_xor.yaml")

# Register the fitness function
pop.set_functions(fitness_function=xor_fitness)

# Evolution loop
last_best_fit = 2.0  # start with a high error (MSE > 1 is clearly non-optimal)

for _ in range(pop.max_generations):
    pop.run_one_generation()
    pop.print_status()

    indiv = pop.best()
    gen = pop.generation_num

    # If a new best individual is found, save a visualization
    if indiv.fitness < last_best_fit:
        last_best_fit = indiv.fitness

        net = indiv.para["brain"].net
        y_pred = [net.calc(x.tolist())[0] for x in X_NORM]

        # Save combined visualization: network structure + fitness curve
        save_combined_net_plot(
            net,
            np.arange(len(X_NORM)),
            Y_TRUE,
            np.array(y_pred),
            f"03_frames/gen_{gen:04d}.png",
            title="Structural Mutation on XOR",
        )

# Final visualization
best = pop.best()
net = best.para["brain"].net
plot_approximation(y_pred, Y_TRUE, title="Best XOR Approximation")
