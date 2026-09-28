from evolib.core.individual import Indiv
from evolib.core.population import Pop
from evolib.operators.selection import selection_tournament, selection_truncation


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
