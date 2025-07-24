# SPDX-License-Identifier: MIT
"""
Initializer for a *simple* feedforward network based on ParaVector.

This module provides initializer functions that interpret a network
structure (layer sizes) as a flat parameter vector. The resulting ParaVector contains
all weights and biases needed for a feedforward network and is fully compatible with
EvoLib mutation, crossover, and adaptation mechanisms.

Use in combination with `NetVector` from `netvector.py` to interpret and evaluate
the parameter vector.

Example:
    cfg.dim = [1, 5, 1]        # 1 input, 5 hidden, 1 output
    cfg.dim_type = "net"       # network structure
    cfg.initializer = "normal_initializer"
    para = normal_initializer_net(cfg)(pop)

    net = NetVector(dim=cfg.dim)
    output = net.forward(input_vector, para.vector)
"""
from typing import TYPE_CHECKING, Callable

import numpy as np

from evolib.config.schema import ComponentConfig
from evolib.representation.netvector import NetVector
from evolib.representation.vector import ParaVector

if TYPE_CHECKING:
    from evolib.core.population import Pop


def normal_initializer_net(cfg: ComponentConfig) -> Callable[["Pop"], ParaVector]:
    """
    Initializes a ParaVector representing a flat parameter vector for a feedforward
    neural network defined by cfg.dim = [input, hidden1, ..., output].

    This does NOT use apply_config(), because that incorrectly flattens dim for net-type
    modules. Instead, the number of parameters is computed explicitly via NetVector.

    Args:
        cfg (ComponentConfig): Must have dim as list[int], and dim_type == "net"

    Returns:
        Callable[[Pop], ParaVector]: Initialization function for a
        ParaVector of correct size
    """

    def init_fn(_: "Pop") -> ParaVector:
        if not isinstance(cfg.dim, list):
            raise ValueError("Expected 'dim' to be list[int] for dim_type='net'")

        # Use NetVector to compute total parameter count
        net_structure = NetVector(dim=cfg.dim, activation=cfg.activation or "tanh")
        n_params = int(net_structure.n_parameters)

        cfg_for_vector = cfg.model_copy()
        cfg_for_vector.dim = n_params
        cfg_for_vector.shape = (n_params,)

        para = ParaVector()
        para.apply_config(cfg_for_vector)

        para.vector = np.random.normal(
            loc=cfg.mean or 0.0,
            scale=cfg.std or 1.0,
            size=n_params,
        )

        return para

    return init_fn
