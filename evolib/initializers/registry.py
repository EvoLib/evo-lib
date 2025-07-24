# SPDX-License-Identifier: MIT
"""
Provides access to registered parameter initializers based on their name and
configuration.

This module dispatches initializer functions based on a string identifier
(e.g. "normal_initializer") and the associated ComponentConfig. Depending on the
'dim_type' field in the configuration, different implementations are returned,
e.g., vector-based or network-based initializers.

Usage:
    init_fn = get_initializer("normal_initializer", cfg)
    para = init_fn(pop)
"""

from evolib.config.schema import ComponentConfig

# Initializers for neural nets
from evolib.initializers.net_initializers import (
    normal_initializer_net,
)

# Initializers for flat vectors
from evolib.initializers.vector_initializers import (
    fixed_initializer,
)
from evolib.initializers.vector_initializers import (
    normal_initializer as normal_initializer_vector,
)
from evolib.initializers.vector_initializers import (
    random_initializer,
    vector_adaptive_initializer,
    zero_initializer,
)
from evolib.interfaces.types import ParaInitializer

# Future: Initializers for other structured types (not yet implemented)
# from evolib.initializers.tensor_initializers import normal_initializer_tensor
# from evolib.initializers.block_initializers import normal_initializer_blocks


def get_initializer(name: str, cfg: ComponentConfig) -> ParaInitializer:
    """
    Returns the appropriate initializer function based on name and configuration.

    Args:
        name (str): The name of the initializer, e.g. "normal_initializer"
        cfg (ComponentConfig): The configuration of the module, including dim_type

    Returns:
        ParaInitializer: A callable that takes a Pop and returns a ParaBase instance

    Raises:
        ValueError: If the name is unknown
        NotImplementedError: If the dim_type is not supported for the given name
    """

    if name == "normal_initializer":
        if cfg.dim_type == "flat":
            return normal_initializer_vector(cfg)
        elif cfg.dim_type == "net":
            return normal_initializer_net(cfg)
        elif cfg.dim_type == "tensor":
            raise NotImplementedError(
                "normal_initializer for dim_type 'tensor'" "is not implemented yet."
            )
        elif cfg.dim_type == "blocks":
            raise NotImplementedError(
                "normal_initializer for dim_type 'blocks'" "is not implemented yet."
            )
        elif cfg.dim_type == "grouped":
            raise NotImplementedError(
                "normal_initializer for dim_type 'grouped'" "is not implemented yet."
            )
        else:
            raise ValueError(f"Unknown dim_type: {cfg.dim_type}")

    elif name == "random_initializer":
        return random_initializer(cfg)

    elif name == "zero_initializer":
        return zero_initializer(cfg)

    elif name == "fixed_initializer":
        return fixed_initializer(cfg)

    elif name == "vector_adaptive":
        return vector_adaptive_initializer(cfg)

    raise ValueError(f"Unknown initializer '{name}'")
