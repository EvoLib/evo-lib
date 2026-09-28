from evolib import Indiv, OptimizationDirection, Pop


def _population_with_fitnesses(config_path: str, fitnesses: list[float]) -> Pop:
    pop = Pop(config_path=config_path)
    pop.indivs = []

    for fitness in fitnesses:
        indiv = Indiv()
        indiv.fitness = fitness
        pop.add_indiv(indiv)

    return pop


def test_optimization_direction_is_loaded_from_config() -> None:
    minimize_pop = Pop(config_path="./tests/configs/population.yaml")
    maximize_pop = Pop(config_path="./tests/configs/population_maximize.yaml")

    assert minimize_pop.optimization_direction == OptimizationDirection.MINIMIZE
    assert maximize_pop.optimization_direction == OptimizationDirection.MAXIMIZE


def test_best_and_elites_respect_maximization() -> None:
    pop = _population_with_fitnesses(
        "./tests/configs/population_maximize.yaml", [1.0, 5.0, 3.0]
    )
    pop.num_elites = 2

    assert pop.best().fitness == 5.0
    assert [indiv.fitness for indiv in pop.get_elites()] == [5.0, 3.0]


def test_statistics_respect_maximization() -> None:
    pop = _population_with_fitnesses(
        "./tests/configs/population_maximize.yaml", [1.0, 5.0, 3.0]
    )

    pop.update_statistics()

    assert pop.best_fitness == 5.0
    assert pop.worst_fitness == 1.0
