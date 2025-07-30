# SPDX-License-Identifier: MIT
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, model_validator

from evolib.config.component_registry import get_component_config_class
from evolib.interfaces.enums import (
    EvolutionStrategy,
    ReplacementStrategy,
    SelectionStrategy,
)


class EvolutionConfig(BaseModel):
    strategy: EvolutionStrategy


class SelectionConfig(BaseModel):
    strategy: SelectionStrategy
    num_parents: Optional[int] = None
    tournament_size: Optional[int] = None
    exp_base: Optional[float] = None
    fitness_maximization: Optional[bool] = False


class ReplacementConfig(BaseModel):
    strategy: ReplacementStrategy = Field(
        ..., description="Replacement strategy to use for survivor selection."
    )
    num_replace: Optional[int] = None
    temperature: Optional[float] = None


class FullConfig(BaseModel):
    """
    Main configuration object for an evolutionary run.

    Includes all meta-settings (evolution, selection, replacement) and the modules
    dictionary, which is resolved into typed ComponentConfigs based on their dim_type.
    """

    parent_pool_size: int
    offspring_pool_size: int
    max_generations: int
    max_indiv_age: int = 0
    num_elites: int

    modules: Dict[str, Any]

    evolution: Optional[EvolutionConfig] = None
    selection: Optional[SelectionConfig] = None
    replacement: Optional[ReplacementConfig] = None

    @model_validator(mode="before")
    @classmethod
    def resolve_component_configs(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Replace each module entry with the appropriate ComponentConfig class, based
        on the declared dim_type."""
        raw_modules = data.get("modules", {})
        resolved = {}
        for name, cfg in raw_modules.items():
            type_name = cfg.get("type", "vector")
            cfg_cls = get_component_config_class(type_name)
            resolved[name] = cfg_cls(**cfg)
        data["modules"] = resolved
        return data
