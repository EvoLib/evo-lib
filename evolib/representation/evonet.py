"""
EvoLib wrapper for EvoNet.

Implements the ParaBase interface for use within EvoLib's evolutionary pipeline.
Supports mutation, crossover, vector conversion, and configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from evonet.core import Nnet
from evonet.mutation import mutate_biases, mutate_weights

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

    def apply_config(self, cfg: ComponentConfig) -> None:
        # load net structure from cfg.dim or similar
        # NOTE: This will be extended in Phase 3
        pass

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
