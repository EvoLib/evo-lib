import matplotlib.pyplot as plt
import numpy as np

from evolib import Indiv, Pop, mse_loss, save_combined_net_plot

# XOR-Daten
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([0, 1, 1, 0])

# Normalisierung (optional, da EvoNet i.d.R. robust)
X_NORM = X.astype(float)
Y_TRUE = Y.astype(float)


# Fitnessfunktion für XOR
def xor_fitness(indiv: Indiv) -> None:
    net = indiv.para["brain"]
    predictions = [net.calc(x.tolist())[0] for x in X_NORM]
    indiv.fitness = mse_loss(Y_TRUE, np.array(predictions))


# Konfiguration laden
pop = Pop(config_path="configs/03_structural_xor.yaml")
pop.set_functions(fitness_function=xor_fitness)

last_best_fit = 2.0
for _ in range(pop.max_generations):
    pop.run_one_generation()
    pop.print_status()

    indiv = pop.best()
    gen = pop.generation_num

    if indiv.fitness < last_best_fit:
        last_best_fit = indiv.fitness

        net = indiv.para["brain"].net
        y_pred = [net.calc(x.tolist())[0] for x in X_NORM]

        # Kombinierte Netzstruktur + Fit speichern
        save_combined_net_plot(
            net,
            np.arange(len(X_NORM)),
            Y_TRUE,
            np.array(y_pred),
            f"03_frames/gen_{gen:04d}.png",
            title="Structural Mutation on XOR",
        )
# Endvisualisierung
best = pop.best()
net = best.para["brain"].net

plt.title("Best XOR Approximation")
plt.plot(Y_TRUE, label="Target", marker="o")
plt.plot([net.calc(x.tolist())[0] for x in X_NORM], label="Best Net", marker="x")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
