from evolib.core.population import Pop


def test_population_initialization() -> None:
    pop = Pop(config_path="./tests/population.yaml")
    assert hasattr(pop, "offspring_pool_size")
    assert isinstance(pop.indivs, list)
