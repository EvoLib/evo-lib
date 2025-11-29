# SPDX-License-Identifier: MIT

import numpy as np
from evonet.core import Nnet
from evonet.mutation import (
    add_random_connection,
    add_random_neuron,
    remove_random_connection,
    remove_random_neuron,
)

from evolib.config.base_component_config import StructuralMutationConfig
from evolib.interfaces.enum_helpers import resolve_recurrent_kinds


def mutate_structure(net: Nnet, cfg: StructuralMutationConfig) -> bool:
    """
    Applies structural mutation operators to the EvoNet.

    Returns
    -------
    bool
        True if a significant topological mutation occurred.
    """

    structure_mutated = False

    # Collect all eligible mutation types (based on probability)
    ops = []
    if (
        cfg.add_connection is not None
        and cfg.add_connection.probability is not None
        and np.random.rand() < cfg.add_connection.probability
    ):
        ops.append("add_connection")
    if (
        cfg.remove_connection is not None
        and cfg.remove_connection.probability is not None
        and np.random.rand() < cfg.remove_connection.probability
    ):
        ops.append("remove_connection")
    if (
        cfg.add_neuron is not None
        and cfg.add_neuron.probability is not None
        and np.random.rand() < cfg.add_neuron.probability
    ):
        ops.append("add_neuron")
    if (
        cfg.remove_neuron is not None
        and cfg.remove_neuron.probability is not None
        and np.random.rand() < cfg.remove_neuron.probability
    ):
        ops.append("remove_neuron")

    # Nothing triggered
    if not ops:
        return False

    # Choose one mutation type to apply
    op = np.random.choice(ops)

    # Add Connection
    if op == "add_connection":
        add_cfg = cfg.add_connection
        if add_cfg is not None:
            if (
                cfg.topology.max_connections is None
                or len(net.get_all_connections()) < cfg.topology.max_connections
            ):
                allowed_kinds = resolve_recurrent_kinds(cfg.topology.recurrent)
                for _ in range(np.random.randint(1, add_cfg.max + 1)):
                    if add_random_connection(
                        net,
                        allowed_recurrent=allowed_kinds,
                        connection_init=add_cfg.init,
                    ):
                        structure_mutated = True

    # Remove Connection
    elif op == "remove_connection":
        rem_cfg = cfg.remove_connection
        if rem_cfg is not None:
            for _ in range(np.random.randint(1, rem_cfg.max + 1)):
                if remove_random_connection(net):
                    structure_mutated = True

    # Add Neuron
    elif op == "add_neuron":
        addn_cfg = cfg.add_neuron
        if cfg.topology.max_connections is not None:
            max_connections = max(0, cfg.topology.max_connections - net.num_weights)
        else:
            max_connections = 2**63 - 1

        if addn_cfg is not None:
            if (
                cfg.topology.max_neurons is None
                or net.num_hidden < cfg.topology.max_neurons
            ):
                if add_random_neuron(
                    net=net,
                    activations=addn_cfg.activations_allowed,
                    connection_init=addn_cfg.init,
                    connection_scope=cfg.topology.connection_scope,
                    connection_density=addn_cfg.init_connection_ratio,
                    max_connections=max_connections,
                ):
                    structure_mutated = True

    # Remove Neuron
    elif op == "remove_neuron":
        remn_cfg = cfg.remove_neuron
        if remn_cfg is not None:
            if remove_random_neuron(net):
                structure_mutated = True
    return structure_mutated
