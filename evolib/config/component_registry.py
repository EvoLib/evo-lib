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
    Resolves the appropriate ComponentConfig class based on the 'type' field defined in
    a module configuration.

    Args:
        type_name (str): The value of cfg["type"]

    Returns:
        Type[BaseModel]: A Pydantic config class

    Raises:
        ValueError: If no matching ComponentConfig is registered
    """
    try:
        return _COMPONENT_MAP[type_name]
    except KeyError:
        raise ValueError(f"Unknown config type: '{type_name}'")
