# SPDX-License-Identifier: MIT
"""
Initializers for structured EvoNet networks using EvoNet.

These initializers convert a module configuration with `type: evonet` into a fully
initialized EvoNet instance.
"""

from evolib.config.schema import FullConfig
from evolib.representation.evonet import EvoNet


def initializer_normal_evonet(config: FullConfig, module: str) -> EvoNet:
    """
    Initializes a EvoNet (EvoNet-based neural network) from config.

    Args:
        config (FullConfig): Full experiment configuration
        module (str): Name of the module (e.g. "brain")

    Returns:
        EvoNet: Initialized EvoNet representation
    """
    para = EvoNet()
    cfg = config.modules[module].model_copy(deep=True)
    para.apply_config(cfg)
    return para
