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
        """
        Resolves raw module dictionaries into typed ComponentConfig objects.

        Each entry in `modules` is a plain dictionary (parsed from YAML).
        This validator inspects the `type` field of each entry (e.g. "vector", "evonet")
        and replaces the dictionary with the corresponding Pydantic ComponentConfig
        subclass (e.g. VectorComponentConfig, EvoNetComponentConfig).

        This ensures that after validation, `FullConfig.modules` always contains
        typed config objects instead of untyped dicts.

        Args:
            data (dict[str, Any]): Raw configuration dictionary provided to FullConfig.

        Returns:
            dict[str, Any]: The updated configuration dictionary, where `modules`
            contains ComponentConfig instances instead of dicts.
        """
        # Extract raw module configs (untyped dicts, e.g. from YAML)
        raw_modules = data.get("modules", {})

        resolved = {}
        for name, cfg in raw_modules.items():
            # Fallback: if no 'type' provided, assume "vector"
            type_name = cfg.get("type", "vector")

            # Select the correct Pydantic config class for this type
            cfg_cls = get_component_config_class(type_name)

            # Instantiate the config class with the provided dictionary
            resolved[name] = cfg_cls(**cfg)

        # Replace raw dicts with validated ComponentConfig objects
        data["modules"] = resolved
        return data
