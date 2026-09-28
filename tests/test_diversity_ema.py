from evolib import Indiv, Pop


def _population_with_fitnesses(fitnesses: list[float]) -> Pop:
    pop = Pop(config_path="./tests/configs/population.yaml")
    pop.indivs = []

    for fitness in fitnesses:
        indiv = Indiv()
        indiv.fitness = fitness
        pop.add_indiv(indiv)

    return pop


def test_diversity_ema_starts_with_first_measured_diversity() -> None:
    pop = _population_with_fitnesses([1.0, 3.0, 5.0])

    assert pop.diversity_ema is None

    pop.update_statistics()

    assert pop.diversity_ema == pop.diversity


def test_reset_clears_diversity_ema() -> None:
    pop = _population_with_fitnesses([1.0, 3.0, 5.0])
    pop.update_statistics()

    pop.reset()

    assert pop.diversity_ema is None
