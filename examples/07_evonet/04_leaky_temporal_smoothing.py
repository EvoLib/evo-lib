# SPDX-License-Identifier: MIT
"""
Example 07-04 – Leaky Temporal Smoothing (Neuron Dynamics)

Show a clear, mechanistically correct advantage of leaky neuron dynamics:
- Leaky neurons integrate input over time (low-pass / exponential smoothing).
- The task requires temporal integration under high-frequency noise.
- Recurrent connections are intentionally not used.

Input: noisy binary signal x[t] in {0,1}
Target: exponential moving average (EMA):
    y[t] = alpha * x[t] + (1 - alpha) * y[t-1]

We compare two configs:
- standard dynamics
- leaky dynamics (typically only in hidden layer)

Fitness
-------
MSE between prediction and EMA target, ignoring a short warmup window.
Lower is better.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evolib import Indiv, Pop

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
CFG_STANDARD = CONFIG_DIR / "04_temporal_smoothing_standard.yaml"
CFG_LEAKY = CONFIG_DIR / "04_temporal_smoothing_leaky.yaml"

# Dataset seed
DATA_SEED = 42

# Time series settings
SEQ_LEN = 400
BASE_HOLD_MIN = 20
BASE_HOLD_MAX = 40
FLIP_PROB = 0.18  # higher -> leaky advantage becomes more obvious

# EMA target
EMA_ALPHA = 0.12  # smaller -> longer memory

# Ignore first steps (net state / EMA start-up)
WARMUP_STEPS = 10


def make_noisy_piecewise_constant_signal(
    *,
    length: int,
    base_hold_min: int,
    base_hold_max: int,
    flip_prob: float,
    seed: int,
) -> np.ndarray:
    """
    Create a noisy binary signal x[t] in {0,1} with slow underlying changes.

    Steps:
    1) Create a clean base signal that stays constant for a random "hold" duration.
       Example: 00000011111100001111...
    2) Flip individual samples with probability `flip_prob` to create noise.

    This yields a signal with low-frequency structure + high-frequency disturbances.
    """
    rng = np.random.default_rng(seed)

    # Build clean base (piecewise constant)
    base = np.zeros(length, dtype=float)

    t = 0
    current = float(rng.integers(0, 2))  # 0.0 or 1.0
    while t < length:
        hold = int(rng.integers(base_hold_min, base_hold_max + 1))
        end = min(length, t + hold)

        base[t:end] = current

        # Toggle 0 <-> 1 for the next segment
        current = 1.0 - current
        t = end

    # Inject noise via random flips
    flip_mask = rng.random(length) < flip_prob
    x = base.copy()
    x[flip_mask] = 1.0 - x[flip_mask]

    return x


def compute_ema(x: np.ndarray, *, alpha: float) -> np.ndarray:
    """
    Exponential moving average (EMA) target.

    y[0] = x[0] y[t] = alpha*x[t] + (1-alpha)*y[t-1]
    """
    y = np.zeros_like(x, dtype=float)
    if x.size == 0:
        return y

    y[0] = float(x[0])
    for t in range(1, len(x)):
        y[t] = alpha * float(x[t]) + (1.0 - alpha) * float(y[t - 1])

    return y


# Build dataset
x_seq = make_noisy_piecewise_constant_signal(
    length=SEQ_LEN,
    base_hold_min=BASE_HOLD_MIN,
    base_hold_max=BASE_HOLD_MAX,
    flip_prob=FLIP_PROB,
    seed=DATA_SEED,
)
y_target = compute_ema(x_seq, alpha=EMA_ALPHA)


def fitness_temporal_smoothing(indiv: Indiv) -> None:
    """Fitness = MSE(pred, target) after warmup."""

    net = indiv.para["nnet"].net
    net.reset(full=True)

    preds = np.zeros_like(x_seq, dtype=float)

    for t in range(len(x_seq)):
        out = float(net.calc([float(x_seq[t])])[0])
        preds[t] = out

    # Output is expected in [0,1] (e.g. sigmoid). Clip defensively.
    preds = np.clip(preds, 0.0, 1.0)

    start = int(WARMUP_STEPS)
    if start >= len(y_target):
        indiv.fitness = float("inf")
        indiv.extra_metrics["mse"] = float("inf")
        return

    mse = float(np.mean((y_target[start:] - preds[start:]) ** 2))
    indiv.fitness = mse
    indiv.extra_metrics["mse"] = mse


def eval_best_prediction(best: Indiv) -> np.ndarray:
    """Return the best individual's predicted time series."""
    net = best.para["nnet"].net
    net.reset(full=True)

    preds = np.zeros_like(x_seq, dtype=float)
    for t in range(len(x_seq)):
        preds[t] = float(net.calc([float(x_seq[t])])[0])

    return np.clip(preds, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Run + plot
# ---------------------------------------------------------------------------


def run_config(
    config_path: Path, label: str, verbosity: int = 1
) -> tuple[np.ndarray, float]:
    """Run evolution for one config and return (best_pred, best_mse)."""
    pop = Pop(str(config_path), fitness_function=fitness_temporal_smoothing)
    pop.run(verbosity=verbosity)

    best = pop.best()
    best_pred = eval_best_prediction(best)

    mse = best.extra_metrics.get("mse")
    if mse is None:
        assert best.fitness is not None
        mse = best.fitness
    best_mse = float(mse)

    print(f"[{label}] best_mse={best_mse:.6f}")

    return best_pred, best_mse


def plot_results(
    *,
    y_std: np.ndarray,
    y_leaky: np.ndarray,
    mse_std: float,
    mse_leaky: float,
) -> None:
    """Plot input, target EMA, and predictions."""
    t = np.arange(len(x_seq))

    plt.figure()
    plt.plot(t, x_seq, label="input x[t] (noisy)")
    plt.plot(t, y_target, label=f"target EMA (alpha={EMA_ALPHA:.2f})")
    plt.plot(t, y_std, label=f"best pred (standard) mse={mse_std:.4f}")
    plt.plot(t, y_leaky, label=f"best pred (leaky) mse={mse_leaky:.4f}")
    plt.xlabel("timestep")
    plt.ylabel("value")
    plt.title("Temporal smoothing: standard vs leaky neuron dynamics")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    # Run standard vs leaky (same dataset, separate evolutionary runs)
    y_std, mse_std = run_config(CFG_STANDARD, label="standard", verbosity=1)
    y_leaky, mse_leaky = run_config(CFG_LEAKY, label="leaky", verbosity=1)

    plot_results(y_std=y_std, y_leaky=y_leaky, mse_std=mse_std, mse_leaky=mse_leaky)


if __name__ == "__main__":
    main()
