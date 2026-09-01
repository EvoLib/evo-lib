# SPDX-License-Identifier: MIT
"""EvoLib adapters for EvoNet topology initializers."""

from typing import Literal

from evolib.config.evonet_component_config import EvoNetComponentConfig
from evolib.config.schema import FullConfig
from evolib.representation.evonet import EvoNet

EvoNetInitializer = Literal["default", "unconnected", "identity"]


def _initialize_evonet(
    config: FullConfig,
    module: str,
    initializer: EvoNetInitializer,
) -> EvoNet:
    cfg = config.modules[module]
    if not isinstance(cfg, EvoNetComponentConfig):
        raise TypeError(f"Module '{module}' is not an EvoNet module")

    # Keep the direct initializer functions deterministic even if called with a
    # component config that specifies another preset.
    cfg = cfg.model_copy(update={"initializer": initializer})
    return EvoNet.from_config(cfg)


def initializer_default_evonet(config: FullConfig, module: str) -> EvoNet:
    """Initialize an EvoNet using the default topology preset."""
    return _initialize_evonet(config, module, "default")


def initializer_unconnected_evonet(config: FullConfig, module: str) -> EvoNet:
    """Initialize an EvoNet without connections."""
    return _initialize_evonet(config, module, "unconnected")


def initializer_identity_evonet(config: FullConfig, module: str) -> EvoNet:
    """Initialize an EvoNet using the identity-like topology preset."""
    return _initialize_evonet(config, module, "identity")
