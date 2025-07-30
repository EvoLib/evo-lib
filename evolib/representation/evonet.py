"""
EvoLib wrapper for EvoNet.

Implements the ParaBase interface for use within EvoLib's evolutionary pipeline.
Supports mutation, crossover, vector conversion, and configuration.
"""

import random
from typing import TYPE_CHECKING

import numpy as np
from evonet.connection import Connection
from evonet.core import Nnet
from evonet.mutation import mutate_biases, mutate_weights
from evonet.neuron import Neuron
from evonet.types import NeuronRole

from evolib.representation.base import ParaBase

if TYPE_CHECKING:
    from evolib.config.schemas import ComponentConfig


class ParaNnet(ParaBase):
    """
    ParaBase wrapper for EvoNet.

    Provides mutation, crossover, and vector I/O for integration with EvoLib.
    """

    def __init__(self) -> None:
        self.net = Nnet()

    def apply_config(self, cfg: "ComponentConfig") -> None:
        dim = cfg.dim
        act = cfg.activation or "tanh"
        w_min, w_max = getattr(cfg, "weight_bounds", (-1.0, 1.0))
        b_min, b_max = getattr(cfg, "bias_bounds", (-0.5, 0.5))

        prev_layer: list[Neuron] = []
        for i, num_neurons in enumerate(dim):
            layer: list[Neuron] = []
            role = (
                NeuronRole.INPUT
                if i == 0
                else NeuronRole.OUTPUT if i == len(dim) - 1 else NeuronRole.HIDDEN
            )
            for _ in range(num_neurons):
                bias = random.uniform(b_min, b_max)
                neuron = Neuron(activation=act, bias=bias)
                self.net.add_neuron(neuron, role=role)
                layer.append(neuron)

            # fully connect to previous layer
            if prev_layer:
                for src in prev_layer:
                    for dst in layer:
                        weight = random.uniform(w_min, w_max)
                        conn = Connection(src, dst, weight)
                        self.net.add_connection(conn)

            prev_layer = layer

    def mutate(self) -> None:
        mutate_weights(self.net)
        mutate_biases(self.net)

    def crossover_with(self, partner: ParaBase) -> None:
        # Placeholder – to be implemented in crossover.py
        # NOTE: Will be implementet in Phase 3
        pass

    def get_vector(self) -> np.ndarray:
        """Returns a flat vector of all weights and biases."""
        weights = self.net.get_weights()
        biases = self.net.get_biases()
        return np.concatenate([weights, biases])

    def set_vector(self, vector: np.ndarray) -> None:
        """Restores weights and biases from a flat vector."""
        n_weights = len(self.net.connections)
        self.net.set_weights(vector[:n_weights])
        self.net.set_biases(vector[n_weights:])

    def get_status(self) -> dict:
        return {
            "neurons": len(self.net.neurons),
            "connections": len(self.net.connections),
        }

    def print_status(self) -> None:
        print(
            f"[ParaNnet] neurons: {len(self.net.neurons)} "
            f"connections: {len(self.net.connections)}"
        )
