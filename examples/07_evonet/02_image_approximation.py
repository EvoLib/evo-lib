"""
EvoNet learns a tiny image from (x, y) coordinates.

This script saves frames only when the best fitness improves (optional).
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from evolib import Indiv, Pop, mse_loss


# Target & grid construction
def make_target(size: int = 6) -> np.ndarray:
    """Return a (H, W) float32 array in [0, 1] as target image."""

    y = np.linspace(-1, 1, size).reshape(-1, 1)
    x = np.linspace(-1, 1, size).reshape(1, -1)
    r = np.sqrt(x * x + y * y)
    ring = np.clip(1.0 - 6.0 * np.abs(r - 0.6), 0.0, 1.0)
    img = np.clip(0.6 * ring, 0.0, 1.0).astype(np.float32)
    return img


def make_coords(size: int) -> np.ndarray:
    """Return grid coordinates normalized to [-1, 1], shape: (H*W, 2)."""
    xs = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)


# Prediction & visualization
def predict_image(indiv: Indiv, coords: np.ndarray, size: int) -> np.ndarray:
    """Map all coords through the network and return a clipped (H, W) image in [0,
    1]."""

    net = indiv.para["nnet"]
    preds = [net.calc(xy.tolist())[0] for xy in coords]
    img = np.array(preds, dtype=np.float32).reshape(size, size)
    return np.clip(img, 0.0, 1.0)


def save_frame(
    path: str, target: np.ndarray, pred: np.ndarray, gen: int, fitness: float
) -> None:
    """Save side-by-side comparison of target and current prediction."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(6, 3), dpi=120)

    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(target, vmin=0.0, vmax=1.0, interpolation="nearest")
    ax1.set_title("Target")
    ax1.axis("off")

    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(pred, vmin=0.0, vmax=1.0, interpolation="nearest")
    ax2.set_title(f"Pred @ gen {gen}\nMSE={fitness:.4f}")
    ax2.axis("off")

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


# Fitness
def my_fitness(indiv: Indiv) -> None:

    pred_img = predict_image(indiv, coords, img_size)
    indiv.fitness = mse_loss(target, pred_img)


def on_improvement(pop: Pop) -> None:
    best = pop.best()
    assert best.fitness is not None
    pred_img = predict_image(best, coords, img_size)
    save_frame(
        path=(f"./02_frames/gen_{pop.generation_num:04d}.png"),
        target=target,
        pred=pred_img,
        gen=pop.generation_num,
        fitness=float(best.fitness),
    )


# Main
img_size: int = 6
target = make_target(size=img_size)
coords = make_coords(img_size)

pop = Pop(
    config_path="configs/02_image_approximation.yaml", fitness_function=my_fitness
)

pop.run(on_improvement=on_improvement)
