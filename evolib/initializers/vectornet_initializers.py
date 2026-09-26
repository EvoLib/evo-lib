# SPDX-License-Identifier: MIT
"""VectorNet initializer adapters delegating to VectorNet.from_config()."""

from evolib.config.schema import FullConfig
from evolib.config.vectornet_component_config import VectorNetComponentConfig
from evolib.representation.vectornet import VectorNet


def initializer_normal_vectornet(config: FullConfig, module: str) -> VectorNet:
    """Initialize a VectorNet using a normal distribution."""
    cfg = config.modules[module]
    if not isinstance(cfg, VectorNetComponentConfig):
        raise TypeError(f"Module '{module}' is not a VectorNet module")

    if cfg.initializer != "normal":
        raise ValueError(
            f"Module '{module}' uses initializer {cfg.initializer!r}; "
            "expected 'normal'."
        )

    return VectorNet.from_config(cfg)
