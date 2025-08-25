# SPDX-License-Identifier: MIT

import numpy as np
from evonet.core import Nnet
from evonet.enums import NeuronRole, RecurrentKind
from evonet.mutation import (
    add_random_connection,
    add_random_neuron,
    remove_random_connection,
    remove_random_neuron,
    split_connection,
)

from evolib.config.base_component_config import StructuralMutationConfig


def resolve_recurrent_kinds(preset: str | None) -> set[RecurrentKind]:
    if preset == "none" or preset is None:
        return set()
    elif preset == "direct":
        return {RecurrentKind.DIRECT}
    elif preset == "local":
        return {RecurrentKind.DIRECT, RecurrentKind.LATERAL}
    elif preset == "all":
        return {RecurrentKind.DIRECT, RecurrentKind.LATERAL, RecurrentKind.INDIRECT}
    else:
        raise ValueError(f"Unknown recurrent preset: {preset}")


def mutate_structure(net: Nnet, cfg: StructuralMutationConfig) -> None:
    """Applies structural mutation operators to the EvoNet."""

    if cfg.add_connection and np.random.rand() < cfg.add_connection:
        if cfg.max_edges is None or len(net.get_all_connections()) < cfg.max_edges:
            allowed_kinds = resolve_recurrent_kinds(cfg.recurrent)
            add_random_connection(net, allowed_recurrent=allowed_kinds)

    if cfg.remove_connection and np.random.rand() < cfg.remove_connection:
        remove_random_connection(net)

    if cfg.add_neuron and np.random.rand() < cfg.add_neuron:
        if cfg.max_nodes is None or count_non_input_neurons(net) < cfg.max_nodes:
            add_random_neuron(net)

    if cfg.remove_neuron and np.random.rand() < cfg.remove_neuron:
        remove_random_neuron(net)

    if cfg.split_connection and np.random.rand() < cfg.split_connection:
        if cfg.max_nodes is None or count_non_input_neurons(net) < cfg.max_nodes:
            split_connection(net)


def count_non_input_neurons(net: Nnet) -> int:
    return len([n for n in net.get_all_neurons() if n.role != NeuronRole.INPUT])
