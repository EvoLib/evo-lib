# SPDX-License-Identifier: MIT
"""This module contains functions for loading strategies for selection, mutation,
crossover, and replacement."""

import importlib
from typing import Callable

# Konstanten für Fehlermeldungen
ERR_UNKNOWN_CATEGORY = "Unbekannte Kategorie: {}. Verfügbare Kategorien: {}"
ERR_NO_GET_STRATEGY = "Modul '{}' besitzt keine Funktion 'get_strategy'"
ERR_INVALID_NAME = (
    "Ungültiger Strategiename: {}. Name darf nur alphanumerische "
    "Zeichen und Unterstriche enthalten"
)

MODULE_MAP = {
    "selection": "evolib.selection",
    "mutation": "evolib.mutation",
    "crossover": "evolib.crossover",
    "replacement": "evolib.replacement",
}


def load_strategy(category: str, name: str) -> Callable:
    """
    Dynamisch eine Strategie aus dem entsprechenden Untermodul laden.

    Args:
        category (str): Kategorie der Strategie (z.B. 'selection', 'mutation')
        name (str): Name der zu ladenden Strategie (z.B. 'tournament')

    Returns:
        Callable: Die geladene Strategie-Funktion

    Raises:
        ValueError: Wenn die Kategorie unbekannt ist oder der Name ungültig ist
        AttributeError: Wenn das Modul keine get_strategy-Funktion besitzt
        ImportError: Wenn das Modul nicht importiert werden kann
    """
    # Typüberprüfung
    if not isinstance(category, str) or not isinstance(name, str):
        raise TypeError("Kategorie und Name müssen Strings sein")

    # Validierung der Kategorie
    if category not in MODULE_MAP:
        raise ValueError(ERR_UNKNOWN_CATEGORY.format(category, list(MODULE_MAP.keys())))

    # Validierung des Strategienamens (nur alphanumerische Zeichen und Unterstriche)
    if not name.isidentifier():
        raise ValueError(ERR_INVALID_NAME.format(name))

    # Erstelle den Modulpfad
    module_path = f"{MODULE_MAP[category]}.{name}"

    try:
        module = importlib.import_module(module_path)

        # Prüfe auf get_strategy Funktion
        if not hasattr(module, "get_strategy"):
            raise AttributeError(ERR_NO_GET_STRATEGY.format(module_path))

        strategy = module.get_strategy()
        return strategy

    except ImportError as e:
        raise ImportError(
            f"Konnte Modul '{module_path}' nicht " "importieren: {str(e)}"
        ) from e
