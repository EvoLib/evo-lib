"""
EvoLib wrapper for EvoNet.

Implements the ParaBase interface for use within EvoLib's evolutionary pipeline.
Supports mutation, crossover, vector conversion, and configuration.
"""

from typing import TYPE_CHECKING

import numpy as np
from evonet.activation import random_function_name
from evonet.core import Nnet
from evonet.enums import NeuronRole
from evonet.mutation import mutate_biases, mutate_weights

from evolib.representation.base import ParaBase

if TYPE_CHECKING:
    from evolib.config.schemas import ComponentConfig


class ParaEvoNet(ParaBase):
    """
    ParaBase wrapper for EvoNet.

    Provides mutation, crossover, and vector I/O for integration with EvoLib.
    """

    def __init__(self) -> None:
        self.net = Nnet()

    def apply_config(self, cfg: "ComponentConfig") -> None:
        dim = cfg.dim
        w_min, w_max = getattr(cfg, "weight_bounds", (-1.0, 1.0))
        b_min, b_max = getattr(cfg, "bias_bounds", (-0.5, 0.5))

        if isinstance(cfg.activation, list):
            activations = cfg.activation
        else:
            activations = [cfg.activation] * len(cfg.dim)

        for layer_idx, num_neurons in enumerate(dim):

            activation_name = activations[layer_idx]

            if activation_name == "random":
                activation_name = random_function_name()

            self.net.add_layer()

            if layer_idx == 0:
                # InputLayer
                role = NeuronRole.INPUT
            elif layer_idx == len(dim) - 1:
                # OutputLayer
                role = NeuronRole.OUTPUT
            else:
                # HiddenLayer
                role = NeuronRole.HIDDEN

            self.net.add_neuron(
                count=num_neurons, activation=activation_name, role=role
            )

    def calc(self, input_values: list[float]) -> list[float]:
        return self.net.calc(input_values)

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

    def get_status(self) -> str:
        return self.net

    def print_status(self) -> None:
        print(f"[ParaEvoNet] : {self.net} ")
