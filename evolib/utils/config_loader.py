# SPDX-License-Identifier: MIT
"""Helper module for loading YAML configuration files."""

from enum import Enum
from pathlib import Path
from typing import Type, TypeVar

import yaml

from evolib.config.schema import FullConfig

T = TypeVar("T", bound=Enum)


def load_config(path: str | Path) -> FullConfig:
    """
    Loads and parses a YAML configuration file into a validated FullConfig object.

    Args:
        path (str | Path): Path to YAML file.

    Returns:
        FullConfig: Fully validated configuration object.
    """
    with open(path, "r") as file:
        raw_cfg = yaml.safe_load(file)

    return FullConfig(**raw_cfg)


def get_enum(enum_class: Type[T], value: str, field_name: str) -> T:
    try:
        return enum_class(value)
    except ValueError as e:
        raise ValueError(f"Unknown {field_name} '{value}' in config") from e
