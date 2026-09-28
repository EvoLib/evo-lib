import pytest

from evolib.core.population import Pop, compute_fitness_diversity
from evolib.interfaces.enums import DiversityMethod


def test_population_initialization() -> None:
    pop = Pop(config_path="./tests/configs/population.yaml")
    assert hasattr(pop, "offspring_pool_size")
    assert isinstance(pop.indivs, list)


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        (DiversityMethod.IQR, 2.0),
        (DiversityMethod.STD, pytest.approx(1.632993161855452)),
        (DiversityMethod.VAR, pytest.approx(8.0 / 3.0)),
        (DiversityMethod.RANGE, 4.0),
    ],
)
def test_compute_fitness_diversity(method: DiversityMethod, expected: float) -> None:
    fitnesses = [-2.0, 0.0, 2.0]

    assert compute_fitness_diversity(fitnesses, method=method) == expected
