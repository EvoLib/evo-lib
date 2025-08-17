# SPDX-License-Identifier: MIT
from typing import Type

from pydantic import BaseModel

from evolib.config.evonet_component_config import EvoNetComponentConfig
from evolib.config.vector_component_config import VectorComponentConfig

# Mapping from 'type' field to corresponding ComponentConfig class
_COMPONENT_MAP: dict[str, Type[BaseModel]] = {
    "vector": VectorComponentConfig,
    "evonet": EvoNetComponentConfig,
    # "composite": CompositeConfig,
    # "torch": TorchComponentConfig,
}


def get_component_config_class(type_name: str) -> Type[BaseModel]:
    """
    Returns the appropriate ComponentConfig class for a given module type.

    This function maps a string identifier from a module config (e.g. "vector",
    "evonet") to the corresponding Pydantic ComponentConfig class
    (e.g. VectorComponentConfig, EvoNetComponentConfig).

    It is typically called by FullConfig.resolve_component_configs() to replace raw
    dictionaries with validated, strongly typed config objects.

    Args:
        type_name (str): The value of the "type" field in the module config.

    Returns:
        Type[BaseModel]: The Pydantic ComponentConfig subclass corresponding
        to the requested type.

    Raises:
        ValueError: If no matching ComponentConfig class is registered.
    """
    try:
        # Look up the config class in the registry mapping
        return _COMPONENT_MAP[type_name]
    except KeyError:
        raise ValueError(f"Unknown config type: '{type_name}'")
