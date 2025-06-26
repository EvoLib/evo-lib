from evolib.utils.benchmarks import ackley, rosenbrock, sphere


def test_sphere():
    assert sphere([0, 0, 0]) == 0.0


def test_rosenbrock():
    assert rosenbrock([1, 1, 1]) == 0.0


def test_ackley():
    assert ackley([0, 0, 0]) >= 0
