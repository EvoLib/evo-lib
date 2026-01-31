"""
Example 07-03 – Delay Bit-Sequence Echo (Delay Mutation Showcase)

Given a deterministic bit sequence u[t] in {0,1}, the network must output:
    y[t] = u[t - k]

Key details:
- Warm-up is used so recurrent delay buffers are filled.
- Fitness combines MSE with a small penalty for misclassification.
- Visualization shows input, target, and predictions during evaluation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from evolib import Indiv, Pop, plot_bit_prediction

FRAME_FOLDER = "03_frames"
CONFIG_FILE = "configs/03_delay_bitseq_echo.yaml"

# Echo target
TARGET_DELAY = 5
DELAY_BOUNDS = (1, 10)
WARMUP_STEPS = DELAY_BOUNDS[1]

# Rollout and plot window
SEQ_LEN = 200
EVAL_LEN = 80

# Local, explicit seed for the deterministic input sequence in this example.
BITS_SEED = 1


def make_bits(n: int, seed: int) -> np.ndarray:
    """Create a bit sequence."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=n, endpoint=False).astype(float)


# Build sequence and echo target
input_seq = make_bits(SEQ_LEN, seed=BITS_SEED)
target_seq = np.roll(input_seq, TARGET_DELAY)


def fitness_echo(indiv: Indiv) -> None:
    """
    Evaluate echo performance on the fixed sequence.

    Fitness = MSE + (1-accuracy)/4 Accuracy uses a hard threshold at 0.5.
    """
    net = indiv.para["nnet"].net
    net.reset(full=True)

    total_error = 0.0
    correct = 0
    count = 0

    for t in range(len(input_seq)):
        output = float(net.calc([float(input_seq[t])])[0])

        if t >= WARMUP_STEPS:
            target = float(target_seq[t])

            # MSE
            err = output - target
            total_error += err * err

            # Accuracy
            pred_bit = 1.0 if output > 0.5 else 0.0
            if pred_bit == target:
                correct += 1

            count += 1

    mse = total_error / max(count, 1)
    acc = correct / max(count, 1)

    indiv.extra_metrics = {"accuracy": acc, "mse": mse}
    indiv.fitness = mse + (1.0 - acc) / 4.0


def save_plot(pop: Pop) -> None:
    best = pop.best()
    net = best.para["nnet"].net
    net.reset(full=True)

    # Warmup
    for bit in input_seq[:WARMUP_STEPS]:
        net.calc([float(bit)])

    # Evaluation window
    start = WARMUP_STEPS
    end = min(WARMUP_STEPS + EVAL_LEN, len(input_seq))
    eval_input = input_seq[start:end]
    eval_target = target_seq[start:end]

    y_preds = [float(net.calc([float(bit)])[0]) for bit in eval_input]

    acc = float(best.extra_metrics.get("accuracy", 0.0))
    mse = float(best.extra_metrics["mse"])

    delays = sorted(
        int(c.delay) for c in net.get_all_connections() if c.type.name == "RECURRENT"
    )

    print(
        f"[Gen {pop.generation_num}] best fitness={best.fitness:.5f}, "
        f"acc={acc:.3%}, mse={mse:.5f}, delays={delays}"
    )

    Path(FRAME_FOLDER).mkdir(parents=True, exist_ok=True)

    plot_bit_prediction(
        true_bits=eval_target.astype(int).tolist(),
        pred_values=y_preds,
        input_bits=eval_input.astype(int).tolist(),
        title=(
            f"Bit Echo (k={TARGET_DELAY}, gen={pop.generation_num}, "
            f"MSE={mse:.5f}, Acc={acc:.3%})"
        ),
        pred_name="Echo",
        show_target_raster=True,
        save_path=f"{FRAME_FOLDER}/gen_{pop.generation_num:03d}.png",
    )


if __name__ == "__main__":
    pop = Pop(CONFIG_FILE, fitness_function=fitness_echo)
    pop.run(verbosity=0, on_generation_end=save_plot)

    best = pop.best()
    best.para["nnet"].net.plot("03_delay_bitseq_echo", fillcolors_on=True)
