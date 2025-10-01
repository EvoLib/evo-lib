"""
Example: Recurrent Trading Strategy

This example demonstrates how to evolve a recurrent EvoNet to act as a
simple trading strategy. Unlike function approximation tasks, the network
does not predict exact values but instead performs time-series classification.

Task:
    Predict one of 5 trend classes at time t (strong down … strong up)
    and convert this prediction into a trading action.

Key aspects:
- Network output encodes 5 classes mapped to trading positions.
- Fitness is based on cumulative trading profit (averaged over multiple runs).
- Accuracy is recorded as a secondary metric.
- Visualization shows trade markers on the price series and the resulting equity curve.
"""

from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np

from evolib import (
    Individual,
    Population,
    generate_timeseries,
    resume_or_create,
    save_checkpoint,
)

# Parameters
PATTERN = "trend_switch"
SEQ_LEN = 400
PRED_LEN = 20
WARMUP_STEPS = max(0, PRED_LEN)

EVAL_RUNS = 10  # number of episodes for fitness
PLOT_SEED = 1234  # fixed seed for reproducible plots

FRAME_FOLDER = "06_frames"
CONFIG_FILE = "./configs/06_recurrent_trading.yaml"

# maps class index (0..4) --> trading position: short/flat/long
CLASS_POS = [-1.0, -0.5, 0.0, 0.5, 1.0]


class EpisodeResult(NamedTuple):
    """
    Container for one trading episode’s results.

    Example:
        EpisodeResult(
            profit=0.12,
            accuracy=0.54,
            equity=[0.0, 0.01, 0.03, ...],
            equity_times=[20, 21, 22, ...],
            trades=[(25, 1.02, "open", +1.0), ...]
        )
    """

    profit: float
    accuracy: float
    equity: list[float]  # starts at 0.0
    equity_times: list[int]  # aligns equity[i] to time index equity_times[i]
    trades: list[tuple[int, float, str, float]]  # (t, price, action, signal)


def checkpoint(pop: Population) -> None:
    save_checkpoint(pop, run_name="06_recurrent_trading")


# Label generation
def make_labels(series: np.ndarray, horizon: int) -> np.ndarray:
    """
    Generate 5-class labels for future trend prediction.

    The label encodes the expected average movement over the next `horizon` steps:
        0 = strongly down, 1 = down, 2 = neutral, 3 = up, 4 = strongly up.

    To adjust the sensitivity of the thresholds to the chosen horizon, a scaling
    factor `scale = sqrt(horizon)` is applied. This prevents longer horizons from
    producing disproportionately small or large class differences.

    Args:
        series: Input time series (normalized price values).
        horizon: Number of steps ahead to evaluate the future trend.

    Returns:
        Numpy array of integer class labels for each time step.
    """

    labels = []
    scale = np.sqrt(horizon)
    for t in range(len(series) - horizon):
        future_mean = np.mean(series[t + 1 : t + 1 + horizon])
        diff = (future_mean - series[t]) / max(1e-6, abs(series[t]))
        if diff < -0.02 * scale:
            labels.append(0)
        elif diff < -0.005 * scale:
            labels.append(1)
        elif diff <= 0.005 * scale:
            labels.append(2)
        elif diff <= 0.02 * scale:
            labels.append(3)
        else:
            labels.append(4)
    return np.array(labels)


# Core trading episode
def run_trading_episode(
    indiv: Individual,
    seq: np.ndarray,
    labels: np.ndarray,
    warmup_steps: int,
    collect_trades: bool = False,
    module: str = "brain",
) -> EpisodeResult:
    """
    Run one trading episode with mark-to-market equity.

    Decision at time t applies to interval t --> t+1.
    Equity is recorded at the END of the interval (t+1), so the plot aligns correctly.
    Before the first position is opened, equity stays flat (position == 0).
    No pyramiding: only one open position, reversed or closed on signal change.

    Returns:
        EpisodeResult: profit, accuracy, equity curve, aligned time indices
        and trade marks.
    """

    net = indiv.para[module].net
    net.reset(full=True)

    # Warmup to initialize recurrent state
    for val in seq[:warmup_steps]:
        net.calc([val])

    position = 0.0
    total_profit = 0.0
    total_correct = 0
    total_samples = 0

    equity = [0.0]
    equity_times = [warmup_steps]  # baseline at start of trading window
    trades: list[tuple[int, float, str, float]] = []

    for t in range(warmup_steps, len(seq) - 1):
        price_now = seq[t]
        price_next = seq[t + 1]

        # Prediction at time t (decision for interval t --> t+1)
        output = net.calc([price_now])
        pred_class = int(np.argmax(output))
        signal = CLASS_POS[pred_class]

        # Accuracy at time t
        if t < len(labels) and pred_class == labels[t]:
            total_correct += 1
        total_samples += 1

        # Trading logic
        if position == 0.0 and signal != 0.0:
            position = signal
            if collect_trades:
                trades.append((t, price_now, "open", signal))
        elif position != 0.0:
            if np.sign(signal) != np.sign(position) and signal != 0.0:
                # close old and open new (direction switch)
                if collect_trades:
                    trades.append((t, price_now, "close+open", signal))
                position = signal
            elif signal == 0.0:
                # close position
                if collect_trades:
                    trades.append((t, price_now, "close", 0.0))
                position = 0.0

        # PnL over interval t→t+1; recorded at t+1 (end of interval)
        pnl = position * (price_next - price_now)
        total_profit += pnl
        equity.append(equity[-1] + pnl)
        equity_times.append(t + 1)

    accuracy = total_correct / max(1, total_samples)
    return EpisodeResult(total_profit, accuracy, equity, equity_times, trades)


# Fitness function
def evaluate(indiv: Individual) -> None:
    """Evaluate an individual using multiple trading episodes."""

    total_profit = 0.0
    total_correct = 0.0
    total_samples = 0

    for _ in range(EVAL_RUNS):
        seed = np.random.randint(0, 2**32 - 1)
        full_seq = generate_timeseries(SEQ_LEN + PRED_LEN, pattern=PATTERN, seed=seed)
        # normalize to [0, 1] for trading analogy (avoid negative "prices")
        full_seq = (full_seq + 1) / 2
        input_seq = full_seq[:-PRED_LEN]
        labels = make_labels(input_seq, PRED_LEN)

        res = run_trading_episode(indiv, input_seq, labels, WARMUP_STEPS)
        profit = res.profit
        acc = res.accuracy

        total_profit += profit
        total_correct += acc * len(input_seq)
        total_samples += len(input_seq)

    avg_profit = total_profit / EVAL_RUNS
    accuracy = total_correct / max(1, total_samples)

    indiv.fitness = -avg_profit  # minimize negative profit
    indiv.extra_metrics["accuracy"] = accuracy
    indiv.extra_metrics["profit"] = avg_profit


# Visualization
def save_plot(pop: Population) -> None:
    """
    Plot trades and equity curve for the current best individual.

    A fixed test sequence (generated with PLOT_SEED) is used to ensure that the
    visualizations are reproducible across runs and generations.
    """

    checkpoint(pop)  # save checkpoint

    best = pop.best()

    # Fixed test sequence for reproducible plots
    full_seq = generate_timeseries(SEQ_LEN + PRED_LEN, pattern=PATTERN, seed=PLOT_SEED)
    # normalize to [0, 1] for trading analogy (avoid negative "prices")
    full_seq = (full_seq + 1) / 2
    input_seq = full_seq[:-PRED_LEN]
    labels = make_labels(input_seq, PRED_LEN)

    result = run_trading_episode(
        best, input_seq, labels, WARMUP_STEPS, collect_trades=True
    )
    profit, acc, equity, equity_times, trades = result

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    # Price series
    ax1.plot(input_seq, color="black", linewidth=1.5)
    for t, price, action, signal in trades:
        if "open" in action:
            ax1.scatter(
                t,
                price,
                color="green" if signal > 0 else "red",
                s=80,
                marker="^",
                label="Long open" if signal > 0 else "Short open",
            )
        elif "close" in action:
            ax1.scatter(t, price, color="blue", s=80, marker="v", label="Close")

    ax1.set_title(
        f"Trading Strategy - Test Run (Gen={pop.generation_num:3d}, "
        f"Hitratio={acc:.2f}, Profit={profit:+.3f})"
    )
    handles, labels_unique = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels_unique, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc="upper right")

    # Equity curve aligned to interval end (t+1)
    ax2.plot(equity_times, equity, linewidth=1.5, label="Equity")
    ax2.set_ylabel("Equity")
    ax2.set_xlabel("Time")

    # Formatting niceties
    ax2.yaxis.set_major_formatter("{x:.2f}")
    ax2.axhline(0.0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(f"{FRAME_FOLDER}/gen_{pop.generation_num:03d}.png", dpi=300)
    plt.close()


# Main
if __name__ == "__main__":
    pop = resume_or_create(CONFIG_FILE, fitness_function=evaluate)
    pop.run(verbosity=1, on_generation_end=save_plot)
