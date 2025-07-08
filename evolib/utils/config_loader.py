# SPDX-License-Identifier: MIT
"""Helper module for loading YAML configuration files."""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, Type, TypeVar

import yaml

T = TypeVar("T", bound=Enum)


def load_config(path: str) -> Dict[str, Any]:
    """
    Lädt eine YAML-Datei und gibt den Inhalt als Dictionary zurück.

    Args:
        path: Pfad zur YAML-Konfigurationsdatei.

    Returns:
        Dictionary mit den Konfigurationswerten.

    Raises:
        FileNotFoundError: Wenn die Datei nicht existiert.
        yaml.YAMLError: Wenn ein YAML-Parsing-Fehler auftritt.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Konfigurationsdatei nicht gefunden: {str(file_path)}")

    with file_path.open("r", encoding="utf-8") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Fehler beim Parsen der YAML-Datei: {e}") from e


def get_enum(enum_class: Type[T], value: str, field_name: str) -> T:
    try:
        return enum_class(value)
    except ValueError as e:
        raise ValueError(f"Unknown {field_name} '{value}' in config") from e
