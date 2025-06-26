from evolib.core.individual import Indiv
from evolib.core.population import Pop
from evolib.operators.selection import selection_tournament


def test_selection_tournament_index() -> None:
    # Dummy-Population aufbauen
    pop = Pop(config_path="./tests/population.yaml")
    pop.indivs = []

    for i in range(5):
        indiv = Indiv()
        indiv.fitness = i + 1.0  # Fitness 1.0, 2.0, ..., 5.0
        pop.add_indiv(indiv)

    # Aufruf mit allen erforderlichen Parametern
    selected = selection_tournament(
        pop=pop, num_parents=2, tournament_size=2, fitness_maximization=False
    )

    assert isinstance(selected, list)
    assert len(selected) == 2
    assert all(isinstance(i, Indiv) for i in selected)
