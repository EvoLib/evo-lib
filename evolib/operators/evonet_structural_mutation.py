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

    # Add Connection (minor)
    if cfg.add_connection and np.random.rand() < cfg.add_connection:
        if cfg.max_edges is None or len(net.get_all_connections()) < cfg.max_edges:
            allowed_kinds = resolve_recurrent_kinds(cfg.recurrent)
            add_random_connection(net, allowed_recurrent=allowed_kinds)

    # Remove Connection (minor)
    if cfg.remove_connection and np.random.rand() < cfg.remove_connection:
        remove_random_connection(net)

    # Add Neuron (significant). Uses allowed activations if provided.
    if cfg.add_neuron and np.random.rand() < cfg.add_neuron:
        if cfg.max_nodes is None or count_non_input_neurons(net) < cfg.max_nodes:
            add_random_neuron(net, cfg.activations_allowed)
            structure_mutated = True

    # Remove Neuron (significant)
    if cfg.remove_neuron and np.random.rand() < cfg.remove_neuron:
        remove_random_neuron(net)
        structure_mutated = True

    # Split Connection (significant)
    if cfg.split_connection and np.random.rand() < cfg.split_connection:
        if cfg.max_nodes is None or count_non_input_neurons(net) < cfg.max_nodes:
            split_connection(net)
            structure_mutated = True

    return structure_mutated


def count_non_input_neurons(net: Nnet) -> int:
    """Return the number of neurons excluding input neurons."""
    return len([n for n in net.get_all_neurons() if n.role != NeuronRole.INPUT])
