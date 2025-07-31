# SPDX-License-Identifier: MIT
"""
Initializers for structured EvoNet networks using ParaNnet.

These initializers convert a module configuration with `type: evonet` into a fully
initialized ParaNnet instance.
"""

from evolib.config.schema import FullConfig
from evolib.representation.evonet import ParaNnet


def initializer_normal_evonet(config: FullConfig, module: str) -> ParaNnet:
    """
    Initializes a ParaNnet (EvoNet-based neural network) from config.

    Args:
        config (FullConfig): Full experiment configuration
        module (str): Name of the module (e.g. "brain")

    Returns:
        ParaNnet: Initialized EvoNet representation
    """
    para = ParaNnet()
    cfg = config.modules[module].model_copy(deep=True)
    para.apply_config(cfg)
    return para
