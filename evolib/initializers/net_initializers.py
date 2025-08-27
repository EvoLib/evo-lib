# SPDX-License-Identifier: MIT
"""
Initializers for Vector interpreted as feedforward neural networks.

These initializers use a 'structure: net' config and convert the network architecture
into a flat parameter vector (weights + biases).

Compatible with NetVector for forward evaluation.
"""

import numpy as np

from evolib.config.schema import FullConfig
from evolib.representation.netvector import NetVector
from evolib.representation.vector import Vector


def initializer_normal_net(config: FullConfig, module: str) -> Vector:
    """
    Initializes a Vector representing a feedforward neural network.

    The network structure is defined via dim = [input, hidden, ..., output],
    and encoded using the NetVector interpreter. All parameters are initialized
    from a normal distribution with cfg.mean and cfg.std.

    Args:
        config (FullConfig): Full configuration object
        module (str): Module name (e.g., "brain")

    Returns:
        Vector: Initialized flat vector representing network weights + biases
    """
    cfg = config.modules[module].model_copy(deep=True)
    # cfg = config.modules[module]

    if not isinstance(cfg.dim, list):
        raise ValueError(f"Module '{module}': expected dim as list[int]")

    net = NetVector(dim=cfg.dim, activation=cfg.activation or "tanh")
    n_params = net.n_parameters

    para = Vector()
    para.apply_config(cfg)

    para.vector = np.random.normal(
        loc=cfg.mean or 0.0,
        scale=cfg.std or 1.0,
        size=n_params,
    )
    if para.init_bounds is not None:
        para.vector = np.clip(para.vector, *para.init_bounds)
    elif para.bounds is not None:
        para.vector = np.clip(para.vector, *para.bounds)

    return para
