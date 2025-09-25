"""
Placeholder implementation of ParaBase for uninitialized individuals.

Ensures that every individual always has a ParaBase instance, even before configuration.
All operations (mutation, crossover, apply_config) raise clear errors to prevent
accidental use of uninitialized parameters.
"""

from typing import Any

from evolib.interfaces.types import ModuleConfig
from evolib.representation.base import ParaBase


class ParaDummy(ParaBase):
    """Placeholder ParaBase used for uninitialized individuals."""

    def apply_config(self, cfg: ModuleConfig) -> None:
        raise RuntimeError("DummyPara cannot be configured.")

    def mutate(self) -> None:
        raise RuntimeError("DummyPara cannot be mutated.")

    def print_status(self) -> None:
        print("[DummyPara] (no parameters)")

    def get_status(self) -> str | dict[str, Any]:
        return "<DummyPara: no parameters>"

    def crossover_with(self, partner: "ParaBase") -> None:
        raise RuntimeError("DummyPara cannot participate in crossover.")
