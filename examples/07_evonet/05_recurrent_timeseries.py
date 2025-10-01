"""
Example: Recurrent Time Series Prediction

This example demonstrates how to evolve a recurrent neural network (EvoNet)
to predict future values in a time series. The task is defined as:

    Predict x[t + PRED_LEN] given x[t]

Key aspects:
- The network receives a warmup phase (WARMUP_STEPS) to stabilize its state.
- Fitness is measured as mean squared error (MSE) between predicted and actual values.
- Multiple evaluation runs (EVAL_RUNS) with different seeds are averaged
  to improve robustness.
- During training, intermediate predictions are visualized and saved as frames.
- After evolution, the best network is tested on a new time series
  (generalization test).
"""

import numpy as np

from evolib import (
    Individual,
    Population,
    generate_timeseries,
    plot_approximation,
    resume_or_create,
    save_checkpoint,
)

# Parameters
PATTERN = "trend_switch"
SEQ_LEN = 400
PRED_LEN = 10
WARMUP_STEPS = max(PRED_LEN, 20)
EVAL_RUNS = 10

FRAME_FOLDER = "05_frames"
CONFIG_FILE = "./configs/05_recurrent_timeseries.yaml"

FULL_SEQ = generate_timeseries(SEQ_LEN + PRED_LEN, pattern=PATTERN, seed=1)
CURRENT_SEQ = FULL_SEQ[:-PRED_LEN]
FUTURE_SEQ = FULL_SEQ[PRED_LEN:]


def eval_identity_baseline(input_seq: np.ndarray, target_seq: np.ndarray) -> float:
    """
    Compute the baseline MSE for the identity predictor: y_pred = x[t].
    This gives a lower bound on how good a naive predictor already is.
    Note: This baseline uses x[t] as a prediction for x[t+PRED_LEN].
    """

    # shift by PRED_LEN already handled in target_seq
    pred_future = np.asarray(input_seq[WARMUP_STEPS:])
    y_true = np.asarray(target_seq[WARMUP_STEPS:])
    return float(np.mean((pred_future - y_true) ** 2))


def gen_y_pred(
    indiv: Individual, input_seq: np.ndarray, module: str = "brain"
) -> list[float]:
    """
    Generate predictions for x[t+PRED_LEN] from x[t] using the individual's network.

    Args:
        indiv: Individual containing the network.
        input_seq: Input time series (length = N).
        module: Which module to use from the individual's para (default: "brain").

    Returns:
        List of predictions aligned with target_seq[WARMUP_STEPS:].
    """

    net = indiv.para[module].net
    net.reset(full=True)

    # Warmup phase
    for time_step in range(WARMUP_STEPS):
        net.calc([input_seq[time_step]])

    # Prediction of x[time_step + PRED_LEN] from x[time_step]
    pred_future = []
    for time_step in range(WARMUP_STEPS, len(input_seq)):
        y_pred = net.calc([input_seq[time_step]])[0]
        pred_future.append(y_pred)

    return pred_future


# Fitness: MSE between predicted and actual
def eval_timeseries_fitness(indiv: Individual) -> None:

    total_mse = 0.0
    for _ in range(EVAL_RUNS):
        seed = np.random.randint(0, 2**32 - 1)
        full_seq = generate_timeseries(SEQ_LEN + PRED_LEN, pattern=PATTERN, seed=seed)
        input_seq = full_seq[:-PRED_LEN]
        target_seq = full_seq[PRED_LEN:]

        # Prediction of x[time_step + PRED_LEN] from x[time_step]
        pred_future = gen_y_pred(indiv, input_seq)

        y_true = target_seq[WARMUP_STEPS:]
        total_mse += float(np.mean((pred_future - y_true) ** 2))

    mse = total_mse / EVAL_RUNS

    indiv.fitness = mse
    indiv.extra_metrics["mse"] = mse


def checkpoint(pop: Population) -> None:
    save_checkpoint(pop, run_name="05_recurrent_timeseries")


# Visualization + Checkpoint
def on_generation_end(pop: Population) -> None:

    checkpoint(pop)

    best = pop.best()
    pred_future = gen_y_pred(best, CURRENT_SEQ)

    current_input = CURRENT_SEQ[WARMUP_STEPS:]
    mse_best = float(np.mean((np.array(pred_future) - current_input) ** 2))
    mse_identity = eval_identity_baseline(CURRENT_SEQ, FUTURE_SEQ)

    print(
        f"Gen={pop.generation_num} MSE(pred)={mse_best:.4f}, "
        f"MSE(identity)={mse_identity:.4f}"
    )

    plot_approximation(
        pred_future,
        current_input,
        title=(
            f"Prediction of x[t+{PRED_LEN}] from x[t]\n"
            f"Gen={pop.generation_num} MSE(best)={mse_best:.4f}, "
            f"MSE(identity)={mse_identity:.4f}"
        ),
        pred_label="Predicted Future",
        true_label="Current Input (x[t])",
        show=False,
        show_grid=True,
        save_path=f"{FRAME_FOLDER}/gen_{pop.generation_num:03d}.png",
    )


# Main loop
def main() -> None:
    print(
        "[Start] Depending on the configuration and available resources, this will take"
        "some time ..."
    )
    pop = resume_or_create(
        CONFIG_FILE,
        fitness_function=eval_timeseries_fitness,
        run_name="05_recurrent_timeseries",
    )

    pop.run(verbosity=0, on_generation_end=on_generation_end)

    # ---------------------------------------------------------
    # Optional: Generalization Test
    # ---------------------------------------------------------
    # After evolution, we test the best network on a completely
    # new time series (different seed) with the same pattern.
    # This demonstrates how well the network generalizes beyond
    # the training runs.
    #
    # Note: The per-generation plots already show generalization
    # (since each generation is evaluated on new random series).
    # This final test is mainly for clarity.
    # ---------------------------------------------------------

    best = pop.best()

    test_seq = generate_timeseries(
        len(CURRENT_SEQ) + PRED_LEN, pattern=PATTERN, seed=219
    )
    test_current = test_seq[:-PRED_LEN]
    test_future = test_seq[PRED_LEN:]

    pred_future = gen_y_pred(best, test_current)
    mse_identity = eval_identity_baseline(test_current, test_future)
    mse_pred = np.mean((pred_future - test_future[WARMUP_STEPS:]) ** 2)

    plot_approximation(
        pred_future,
        test_current[WARMUP_STEPS:],
        title=(
            f"Generalization Test (new series)"
            f"MSE(best)={mse_pred:.4f}, "
            f"MSE(identity)={mse_identity:.4f}"
        ),
        pred_label="Predicted Future",
        true_label="Current Input (x[t])",
        show=False,
        save_path=f"{FRAME_FOLDER}/00_Generalization_Test.png",
    )


if __name__ == "__main__":
    main()
