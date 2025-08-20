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
    """
    Top-level evolution policy.

    Holds only the high-level strategy choice; operator-specific behavior lives in the
    Para* representations and operator modules.
    """

    strategy: EvolutionStrategy = Field(
        ..., description="High-level evolution strategy (e.g. (mu_plus_lambda)."
    )


class SelectionConfig(BaseModel):
    """
    Parent selection settings.

    Depending on the selected strategy, only some fields are relevant; the actual
    semantics are implemented in the selection registry.
    """

    strategy: SelectionStrategy = Field(
        ..., description="Parent selection strategy (e.g. tournament, ranking)."
    )
    num_parents: Optional[int] = Field(
        None, description="Optional override for the number of parents to pick."
    )
    tournament_size: Optional[int] = Field(
        None, description="Tournament size for tournament selection."
    )
    exp_base: Optional[float] = Field(
        None, description="Base for exponential ranking selection."
    )
    fitness_maximization: Optional[bool] = Field(
        False, description="If True, higher fitness is considered better."
    )


class ReplacementConfig(BaseModel):
    """
    Survivor replacement (environmental selection) settings.

    Concrete behavior is implemented in the replacement registry.
    """

    strategy: ReplacementStrategy = Field(
        ..., description="Survivor selection strategy (e.g. replace_worst, anneal)."
    )
    num_replace: Optional[int] = Field(
        None, description="How many individuals to replace (strategy-dependent)."
    )
    temperature: Optional[float] = Field(
        None, description="Temperature parameter for annealing-like strategies."
    )


class FullConfig(BaseModel):
    """
    Main configuration model for an evolutionary run.

    Aggregates global run parameters, high-level policies (evolution/selection/
    replacement), and a 'modules' mapping that is resolved into typed ComponentConfigs.

    1) YAML → dict 2) dict → FullConfig(**data) 3) model_validator(mode="before")
    resolves each raw 'modules[name]' dict into a    typed ComponentConfig (e.g.
    VectorComponentConfig, EvoNetComponentConfig).
    """

    # Global run parameters
    parent_pool_size: int = Field(
        ..., description="Number of parents retained in each generation."
    )
    offspring_pool_size: int = Field(
        ..., description="Number of offspring produced per generation."
    )
    max_generations: int = Field(
        ..., description="Maximum number of generations to run."
    )
    max_indiv_age: int = Field(
        0,
        description="Maximum allowed individual age in generations; 0 disables aging.",
    )
    num_elites: int = Field(
        ..., description="Number of elite individuals preserved each generation."
    )

    # Module configs (resolved to typed ComponentConfig instances by the validator)
    modules: Dict[str, Any]

    # High-level policies (optional)
    evolution: Optional[EvolutionConfig] = Field(
        None, description="Global evolution strategy configuration."
    )
    selection: Optional[SelectionConfig] = Field(
        None, description="Parent selection configuration."
    )
    replacement: Optional[ReplacementConfig] = Field(
        None, description="Survivor selection (replacement) configuration."
    )

    @model_validator(mode="before")
    @classmethod
    def resolve_component_configs(cls, data: dict[str, Any]) -> dict[str, Any]:
        """
        Replace raw 'modules[name]' dicts with typed ComponentConfig objects.

        Steps:
          1) Read raw dict from 'modules[name]'
          2) Determine 'type' (default: "vector")
          3) Lookup ComponentConfig class in the registry
          4) Instantiate the typed model with the raw dict

        After this hook, 'modules' contains fully validated Pydantic models.

        Raises
        ------
        ValueError
            If a module 'type' is unknown to the component registry.
        """
        raw_modules = data.get("modules", {})
        resolved: dict[str, Any] = {}

        for name, cfg in raw_modules.items():
            type_name = cfg.get("type", "vector")
            cfg_cls = get_component_config_class(type_name)
            resolved[name] = cfg_cls(**cfg)

        data["modules"] = resolved
        return data
