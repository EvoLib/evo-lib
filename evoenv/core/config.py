# SPDX-License-Identifier: MIT
"""Shared Pydantic base models for EvoEnv configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict


class StrictConfigModel(BaseModel):
    """Base model that rejects unknown configuration fields."""

    model_config = ConfigDict(extra="forbid")


class YamlConfigModel(StrictConfigModel):
    """Base model for strict configurations loaded from YAML files."""

    def to_yaml_dict(self) -> dict[str, object]:
        """Return a YAML-serializable representation of the configuration."""
        return self.model_dump(mode="json")

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        """Load and validate a configuration from a YAML file."""
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as file:
            raw_config = yaml.safe_load(file) or {}

        return cls.model_validate(raw_config)
