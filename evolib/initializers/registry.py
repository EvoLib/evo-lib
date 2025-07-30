# SPDX-License-Identifier: MIT
"""
Provides access to registered parameter initializers based on their name and
configuration.

This module dispatches initializer functions based on a string identifier
(e.g. "normal_initializer") and the associated module name within FullConfig.

Usage:
    init_fn = get_initializer("ininitializer_normal_vector")
    para = init_fn(config, module="brain")
"""

from typing import Any, Callable

from evolib.config.schema import FullConfig

# EvoNet-based initializer
from evolib.initializers.evonet_initializers import initializer_normal_evonet

# NetVector-based initializer
from evolib.initializers.net_initializers import initializer_normal_net

# Vector-based initializers
from evolib.initializers.vector_initializers import (
    initializer_adaptive_vector,
    initializer_fixed_vector,
    initializer_normal_vector,
    initializer_random_vector,
    initializer_zero_vector,
)
from evolib.representation.base import ParaBase
from evolib.representation.composite import ParaComposite

# Typalias for initializer function
InitializerFunction = Callable[[FullConfig, str], ParaBase]


# Registry of known initializer functions
INITIALIZER_REGISTRY: dict[str, InitializerFunction] = {
    "normal_vector": initializer_normal_vector,
    "random_vector": initializer_random_vector,
    "zero_vector": initializer_zero_vector,
    "fixed_vector": initializer_fixed_vector,
    "adaptive_vector": initializer_adaptive_vector,
    "normal_net": initializer_normal_net,
    "normal_evonet": initializer_normal_evonet,
}


def get_initializer(name: str) -> InitializerFunction:
    """
    Returns the initializer function for the given name.

    Args:
        name (str): Identifier of the initializer (must match registry)

    Returns:
        Callable[[FullConfig, str], ParaBase]: Initializer function

    Raises:
        ValueError: If the name is not registered
    """
    if name not in INITIALIZER_REGISTRY:
        raise ValueError(f"Unknown initializer: '{name}'")

    return INITIALIZER_REGISTRY[name]


def build_composite_initializer(config: FullConfig) -> Callable[[Any], ParaComposite]:
    def initializer(_: Any) -> ParaComposite:
        return ParaComposite(
            {
                name: get_initializer(config.modules[name].initializer)(config, name)
                for name in config.modules
            }
        )

    return initializer
