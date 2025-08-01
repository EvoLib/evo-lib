# SPDX-License-Identifier: MIT
"""
Initializers for structured EvoNet networks using ParaEvoNet.

These initializers convert a module configuration with `type: evonet` into a fully
initialized ParaEvoNet instance.
"""

from evolib.config.schema import FullConfig
from evolib.representation.evonet import ParaEvoNet


def initializer_normal_evonet(config: FullConfig, module: str) -> ParaEvoNet:
    """
    Initializes a ParaEvoNet (EvoNet-based neural network) from config.

    Args:
        config (FullConfig): Full experiment configuration
        module (str): Name of the module (e.g. "brain")

    Returns:
        ParaEvoNet: Initialized EvoNet representation
    """
    para = ParaEvoNet()
    cfg = config.modules[module].model_copy(deep=True)
    para.apply_config(cfg)
    return para
