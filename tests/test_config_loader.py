from pathlib import Path

import pytest
import yaml

from evolib.config.schema import FullConfig
from evolib.utils.config_loader import load_config


@pytest.fixture
def minimal_config_yaml(tmp_path: Path) -> Path:
    content = {
        "parent_pool_size": 10,
        "offspring_pool_size": 20,
        "max_generations": 5,
        "max_indiv_age": 0,
        "num_elites": 1,
        "evolution": {"strategy": "mu_plus_lambda"},
        "modules": {
            "main": {
                "type": "vector",
                "dim": 4,
                "initializer": "zero_initializer",
                "bounds": [-1.0, 1.0],
                "mutation": {
                    "strategy": "constant",
                    "strength": 0.1,
                    "probability": 1.0,
                },
                "crossover": {
                    "strategy": "constant",
                    "operator": "blx",
                    "probability": 0.5,
                },
            }
        },
    }
    path = tmp_path / "test_config.yaml"
    with path.open("w") as f:
        yaml.dump(content, f, sort_keys=False)
    return path


def test_config_parses_successfully(minimal_config_yaml: Path) -> None:
    cfg = load_config(minimal_config_yaml)
    assert isinstance(cfg, FullConfig)
    assert cfg.parent_pool_size == 10
    assert "main" in cfg.modules
    module = cfg.modules["main"]
    assert module.dim == 4
    assert module.bounds == (-1.0, 1.0)
    assert module.mutation is not None
    assert module.crossover is not None
    assert module.mutation.strategy.value == "constant"
    assert module.crossover.operator is not None
    assert module.crossover.operator.value == "blx"
