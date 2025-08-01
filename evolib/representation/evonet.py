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

from evolib.interfaces.enums import CrossoverStrategy, MutationStrategy
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

        # Mutationstrategy
        self.mutation_strategy: MutationStrategy | None = None

        # Global Mutationparameter
        self.mutation_strength: float | None = None
        self.mutation_probability: float | None = None
        self.tau: float = 0.0

        # Per-Parameter Mutationparameter
        self.para_mutation_strengths: np.ndarray | None = None
        self.randomize_mutation_strengths: bool | None = None

        # Bounds of parameter (z. B. [-1, 1])
        self.bounds: tuple[float, float] | None = None
        self.init_bounds: tuple[float, float] | None = None

        # Parametervektor
        self.vector: np.ndarray = np.zeros(1)
        self.shape: tuple[int, ...] = (1,)

        # Bounds for mutation (min/max)
        self.min_mutation_strength: float | None = None
        self.max_mutation_strength: float | None = None
        self.min_mutation_probability: float | None = None
        self.max_mutation_probability: float | None = None

        # Diversity based Adaptionfaktors
        self.mutation_inc_factor: float | None = None
        self.mutation_dec_factor: float | None = None
        self.min_diversity_threshold: float | None = None
        self.max_diversity_threshold: float | None = None

        # Crossover
        self.crossover_strategy: CrossoverStrategy | None = None
        self.crossover_probability: float | None = None
        self.min_crossover_probability: float | None = None
        self.max_crossover_probability: float | None = None
        self.crossover_inc_factor: float | None = None
        self.crossover_dec_factor: float | None = None
        self._crossover_fn = None

    def apply_config(self, cfg: "ComponentConfig") -> None:
        w_min, w_max = getattr(cfg, "weight_bounds", (-1.0, 1.0))
        b_min, b_max = getattr(cfg, "bias_bounds", (-0.5, 0.5))

        # Assign dimensions
        self.dim = cfg.dim

        # Mutation
        if cfg.mutation is None:
            raise ValueError("Mutation config is required for ParaEvoNet.")
        self.mutation_strategy = cfg.mutation.strategy

        # Strategy-specific mutation params
        m = cfg.mutation
        if self.mutation_strategy == MutationStrategy.CONSTANT:
            self.mutation_probability = m.probability
            self.mutation_strength = m.strength

        if isinstance(cfg.activation, list):
            activations = cfg.activation
        else:
            activations = [cfg.activation] * len(cfg.dim)

        for layer_idx, num_neurons in enumerate(self.dim):

            activation_name = activations[layer_idx]

            if activation_name == "random":
                activation_name = random_function_name()

            self.net.add_layer()

            if layer_idx == 0:
                # InputLayer
                role = NeuronRole.INPUT
            elif layer_idx == len(self.dim) - 1:
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
        mutate_weights(self.net, std=self.mutation_strength)
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

    def print_graph(
        self,
        name: str,
        engine: str = "neato",
        labels_on: bool = True,
        colors_on: bool = True,
        thickness_on: bool = False,
        fillcolors_on: bool = False,
    ) -> None:
        """
        Prints the graph structure of the EvoNet.

        Args:
            name (str): Output filename (without extension).
            engine (str): Layout engine for Graphviz.
            labels_on (bool): Show edge weights as labels.
            colors_on (bool): Use color coding for edge weights.
            thickness_on (bool): Adjust edge thickness by weight.
            fillcolors_on (bool): Fill nodes with colors by type.
        """
        self.net.print_graph(
            name=name,
            engine=engine,
            labels_on=labels_on,
            colors_on=colors_on,
            thickness_on=thickness_on,
            fillcolors_on=fillcolors_on,
        )
