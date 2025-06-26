from evolib.utils.config_loader import load_config


def test_load_population_config() -> None:
    config = load_config("./tests/population.yaml")
    assert isinstance(config, dict)
    assert "parent_pool_size" in config
