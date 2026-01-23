"""
Example 07-02 – Delay example for EvoNet.

Demonstrates how explicit recurrent delays propagate past inputs through time. This
example is intentionally minimal and serves as a delay semantics test.

Given an input signal u[t], the network outputs u[t−k] using an explicit recurrent
delay. The network does not need to approximate a sine from phase or time. It only
reproduces a past input value, isolating delay semantics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from evolib import Indiv, Pop, mse_loss, plot_approximation

# ---------------------------------------------------------------------
# Configuration (keep in sync with YAML)
# ---------------------------------------------------------------------

CONFIG_PATH = "configs/02_sine_delay.yaml"

DELAY_STEPS = 15  # must match YAML: delay.value
N_SAMPLES = 200
X_RAW = np.linspace(0.0, 2.0 * np.pi, N_SAMPLES)

# Output folders for optional frames
FRAMES_DIR = Path("02_frames")
FRAMES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Build input signal u[t] and delayed target y[t] = u[t-k]
# ---------------------------------------------------------------------

Y_TRUE = np.sin(X_RAW)
Y_TARGET = np.roll(Y_TRUE, DELAY_STEPS)


# ---------------------------------------------------------------------
# Fitness function
# ---------------------------------------------------------------------


def evonet_fitness(indiv: Indiv) -> None:
    """
    Evaluate how well the network reproduces the input from k steps ago.

    We roll out the network sequentially so recurrent delay buffers are used. The first
    DELAY_STEPS are ignored because the delay buffer cannot be filled yet.
    """
    net = indiv.para["nnet"].net

    # Full reset clears delay buffers (important across evaluations)
    net.reset(full=True)

    preds = np.empty_like(Y_TRUE, dtype=float)

    for t in range(len(Y_TRUE)):
        preds[t] = float(net.calc([float(Y_TRUE[t])])[0])

    # Ignore warm-up steps where the delay buffer is not yet filled
    indiv.fitness = mse_loss(Y_TARGET[DELAY_STEPS:], preds[DELAY_STEPS:])


# ---------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------


def on_improvement(pop: Pop) -> None:
    """
    Save a frame on improvements for visual debugging.

    This is optional but very useful to spot buffer/reset/delay issues quickly.
    """
    indiv = pop.best()
    net = indiv.para["nnet"].net
    net.reset(full=True)

    y_preds = np.array([net.calc([float(u)])[0] for u in Y_TRUE], dtype=float)

    out_path = FRAMES_DIR / f"gen_{pop.generation_num:04d}.png"
    plot_approximation(
        y_preds,
        Y_TRUE,
        title=f"Delayed Sine Reconstruction (delay_steps={DELAY_STEPS})",
        pred_marker=None,
        true_marker=None,
        show=False,
        save_path=str(out_path),
        x_vals=X_RAW,
        y_limits=(-1.2, 1.2),
    )


# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    pop = Pop(
        config_path=CONFIG_PATH,
        fitness_function=evonet_fitness,
    )

    pop.run(
        on_improvement=on_improvement,
        verbosity=1,
    )
