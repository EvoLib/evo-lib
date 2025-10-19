# SPDX-License-Identifier: MIT

import numpy as np
from evonet.core import Nnet
from evonet.enums import NeuronRole
from evonet.mutation import (
    add_random_connection,
    add_random_neuron,
    remove_random_connection,
    remove_random_neuron,
    split_connection,
)

from evolib.config.base_component_config import StructuralMutationConfig
from evolib.interfaces.enum_helpers import resolve_recurrent_kinds


def mutate_structure(net: Nnet, cfg: StructuralMutationConfig) -> bool:
    """
    Applies structural mutation operators to the EvoNet.

    Only *significant* topological changes (adding/removing neurons or splitting
    a connection) will set `structure_mutated = True`.
    Simple connection add/remove operations are treated as minor changes and
    do not trigger HELI incubation.

    Returns
    -------
    bool
        True if a significant topological mutation occurred.
    """

    structure_mutated = False

    # Collect all eligible mutation types (based on probability)
    ops = []
    if cfg.add_connection and np.random.rand() < cfg.add_connection:
        ops.append("add_connection")
    if cfg.remove_connection and np.random.rand() < cfg.remove_connection:
        ops.append("remove_connection")
    if cfg.add_neuron and np.random.rand() < cfg.add_neuron:
        ops.append("add_neuron")
    if cfg.remove_neuron and np.random.rand() < cfg.remove_neuron:
        ops.append("remove_neuron")
    if cfg.split_connection and np.random.rand() < cfg.split_connection:
        ops.append("split_connection")

    # Nothing triggered
    if not ops:
        return False

    # Choose one mutation type to apply
    op = np.random.choice(ops)

    # Add Connection (minor)
    if op == "add_connection":
        if cfg.max_edges is None or len(net.get_all_connections()) < cfg.max_edges:
            allowed_kinds = resolve_recurrent_kinds(cfg.recurrent)
            for _ in range(np.random.randint(1, cfg.max_new_connections + 1)):
                add_random_connection(
                    net,
                    allowed_recurrent=allowed_kinds,
                    connection_init=cfg.connection_init,
                )
            structure_mutated = True

    # Remove Connection (minor)
    elif op == "remove_connection":
        for _ in range(np.random.randint(1, cfg.max_removed_connections + 1)):
            remove_random_connection(net)
            structure_mutated = True

    # Add Neuron (significant). Uses allowed activations if provided.
    elif op == "add_neuron":
        if cfg.max_nodes is None or count_non_input_neurons(net) < cfg.max_nodes:
            add_random_neuron(
                net,
                cfg.activations_allowed,
                cfg.connection_init,
                cfg.connection_scope,
                cfg.connection_density,
            )
            structure_mutated = True

    # Remove Neuron (significant)
    elif op == "remove_neuron":
        remove_random_neuron(net)
        structure_mutated = True

    # Split Connection (significant)
    elif op == "split_connection":
        if cfg.max_nodes is None or count_non_input_neurons(net) < cfg.max_nodes:
            split_connection(net)
            structure_mutated = True

    return structure_mutated


def count_non_input_neurons(net: Nnet) -> int:
    """Return the number of neurons excluding input neurons."""
    return len([n for n in net.get_all_neurons() if n.role != NeuronRole.INPUT])
