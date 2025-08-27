"""Approximating sin(x) using a feedforward network defined via EvoNet."""

import matplotlib.pyplot as plt
import numpy as np

from evolib import Indiv, Pop, mse_loss, save_combined_net_plot

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


# Run evolution
pop = Pop(config_path="configs/01_sine_approximation.yaml")
pop.set_functions(fitness_function=evonet_fitness)


last_best_fitness = 2.0
for _ in range(pop.max_generations):

    pop.run_one_generation()
    pop.print_status()

    gen = pop.generation_num

    indiv = pop.best()

    if indiv.fitness < last_best_fitness:
        last_best_fitness = indiv.fitness
        net = indiv.para["nnet"].net
        y_pred = np.array([net.calc([x])[0] for x in X_NORM])

        save_combined_net_plot(
            net, X_RAW, Y_TRUE, y_pred, f"01_frames/gen_{gen:04d}.png"
        )
exit(0)
# Visualize result
best = pop.best()
y_best = [pop.best().para["nnet"].calc([x])[0] for x in X_NORM]

plt.plot(X_RAW, Y_TRUE, label="Target: sin(x)")
plt.plot(X_RAW, y_best, label="Best Approximation", linestyle="--")

plt.title("EvoNet Fit to sin(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
