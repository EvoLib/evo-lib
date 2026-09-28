import numpy as np
import pytest

from evolib.core.individual import Indiv
from evolib.core.population import Pop
from evolib.operators.selection import (
    _calculate_rank_probabilities,
    selection_rank_based,
    selection_tournament,
    selection_truncation,
)


def test_selection_tournament_index() -> None:
    # Build a small test population
    pop = Pop(config_path="./tests/configs/population.yaml")
    pop.indivs = []

    for i in range(5):
        indiv = Indiv()
        indiv.fitness = i + 1.0  # Fitness 1.0, 2.0, ..., 5.0
        pop.add_indiv(indiv)

    # Select two parents
    selected = selection_tournament(pop=pop, num_parents=2, tournament_size=2)

    assert isinstance(selected, list)
    assert len(selected) == 2
    assert all(isinstance(i, Indiv) for i in selected)


def test_selection_truncation_respects_maximization() -> None:
    pop = Pop(config_path="./tests/configs/population_maximize.yaml")
    pop.indivs = []

    for fitness in [1.0, 5.0, 3.0]:
        indiv = Indiv()
        indiv.fitness = fitness
        pop.add_indiv(indiv)

    selected = selection_truncation(pop, num_parents=2)

    assert [indiv.fitness for indiv in selected] == [5.0, 3.0]


def test_linear_rank_probabilities_favor_best_rank() -> None:
    ranks = np.arange(3)

    probabilities = _calculate_rank_probabilities(
        ranks, population_size=3, mode="linear", exp_base=1.0
    )

    assert probabilities[0] > probabilities[1] > probabilities[2]


def test_exponential_rank_probabilities_favor_best_rank() -> None:
    ranks = np.arange(3)

    probabilities = _calculate_rank_probabilities(
        ranks, population_size=3, mode="exponential", exp_base=2.0
    )

    assert probabilities[0] > probabilities[1] > probabilities[2]


def test_exponential_rank_base_one_is_uniform() -> None:
    ranks = np.arange(3)

    probabilities = _calculate_rank_probabilities(
        ranks, population_size=3, mode="exponential", exp_base=1.0
    )

    np.testing.assert_allclose(probabilities, np.full(3, 1.0 / 3.0))


def test_exponential_rank_rejects_base_below_one() -> None:
    pop = Pop(config_path="./tests/configs/population.yaml")

    with pytest.raises(ValueError, match="greater than or equal to 1.0"):
        selection_rank_based(pop, num_parents=1, mode="exponential", exp_base=0.5)
