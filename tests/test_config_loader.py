# tests/test_config_loader.py

from pathlib import Path

import pytest
from pydantic import ValidationError

from evolib.config.schema import FullConfig, MutationStrategy, SelectionStrategy
from evolib.utils.config_loader import load_config


@pytest.fixture
def valid_config_path(tmp_path: Path) -> Path:
    content = """
parent_pool_size: 10
offspring_pool_size: 20
max_generations: 100
max_indiv_age: 10
num_elites: 1

representation:
  type: vector
  dim: 5
  bounds: [-1.0, 1.0]
  initializer: uniform

mutation:
  strategy: adaptive_global
  strength: 0.2
  probability: 0.9

selection:
  strategy: tournament
  tournament_size: 4

replacement:
  strategy: generational
"""
    config_file = tmp_path / "valid_config.yaml"
    config_file.write_text(content)
    return config_file


def test_load_valid_config(valid_config_path: Path) -> None:
    cfg = load_config(valid_config_path)

    assert isinstance(cfg, FullConfig)
    assert cfg.parent_pool_size == 10
    assert cfg.representation.dim == 5
    assert cfg.mutation.strategy == MutationStrategy.ADAPTIVE_GLOBAL
    assert cfg.selection is not None
    assert cfg.selection.strategy == SelectionStrategy.TOURNAMENT
    assert cfg.selection.tournament_size == 4
    assert cfg.replacement is not None
    assert cfg.replacement.strategy.value == "generational"
    assert cfg.representation.bounds == (-1.0, 1.0)


def test_missing_required_fields(tmp_path: Path) -> None:
    """Test that missing required fields raise ValidationError."""
    content = """
mutation:
  strategy: constant
representation:
  type: vector
  dim: 5
  bounds: [-1, 1]
  initializer: uniform
"""
    config_file = tmp_path / "invalid_config.yaml"
    config_file.write_text(content)

    with pytest.raises(ValidationError) as exc_info:
        load_config(config_file)

    assert "parent_pool_size" in str(exc_info.value)
    assert "offspring_pool_size" in str(exc_info.value)


def test_optional_blocks(tmp_path: Path) -> None:
    """Test that config can be loaded without optional blocks like selection or
    evolution."""
    content = """
parent_pool_size: 5
offspring_pool_size: 5
max_generations: 20
max_indiv_age: 0
num_elites: 0

representation:
  type: vector
  dim: 3
  bounds: [-5.0, 5.0]
  initializer: uniform

mutation:
  strategy: constant
"""
    config_file = tmp_path / "minimal_config.yaml"
    config_file.write_text(content)

    cfg = load_config(config_file)
    assert cfg.selection is None
    assert cfg.evolution is None
    assert cfg.crossover is None
