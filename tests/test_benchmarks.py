import numpy as np

from evolib.utils.benchmarks import ackley, rosenbrock, sphere


def test_sphere() -> None:
    assert sphere(np.array([0, 0, 0])) == 0.0


def test_rosenbrock() -> None:
    assert rosenbrock(np.array([1, 1, 1])) == 0.0


def test_ackley() -> None:
    assert ackley(np.array([0, 0, 0])) >= 0
