import pytest
from pydantic import ValidationError

from evolib.utils.config_loader import load_config
from evolib.utils.config_validator import validate_full_config


def test_validate_valid_config() -> None:
    # Beispielhafte gültige YAML-Datei
    cfg = load_config("examples/02_strategies/05_adaptive_global.yaml")
    validated = validate_full_config(cfg)

    # Basisprüfungen
    assert validated.parent_pool_size > 0
    assert validated.mutation.strategy.value == "adaptive_global"
    assert validated.representation.type.value == "vector"
    assert validated.representation.bounds[0] < validated.representation.bounds[1]


def test_validate_invalid_config_missing_fields() -> None:
    # absichtlich fehlerhaftes Config-Dict (z. B. fehlende mutation.strategy)
    cfg = {
        "parent_pool_size": 10,
        "offspring_pool_size": 30,
        "max_generations": 50,
        "representation": {
            "type": "vector",
            "dim": 3,
            "bounds": [-1.0, 1.0],
            "initializer": "uniform",
        },
        "mutation": {
            # "strategy" fehlt!
            "strength": 0.1,
        },
    }

    with pytest.raises(ValidationError) as exc_info:
        validate_full_config(cfg)

    assert "mutation -> strategy" in str(exc_info.value) or "mutation.strategy" in str(
        exc_info.value
    )
