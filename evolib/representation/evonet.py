"""
EvoLib wrapper for EvoNet.

Implements the ParaBase interface for use within EvoLib's evolutionary pipeline.
Supports mutation, crossover, vector conversion, and configuration.
"""

import numpy as np
from evonet.activation import random_function_name
from evonet.core import Nnet
from evonet.enums import NeuronRole
from evonet.mutation import mutate_biases, mutate_weights

from evolib.config.evonet_component_config import EvoNetComponentConfig
from evolib.interfaces.enums import MutationStrategy
from evolib.interfaces.types import ModuleConfig
from evolib.representation.base import ParaBase
from evolib.representation.evo_params import EvoControlParams


class ParaEvoNet(ParaBase):
    """
    ParaBase wrapper for EvoNet.

    Provides mutation, crossover, and vector I/O for integration with EvoLib.
    """

    def __init__(self) -> None:
        self.net = Nnet()

        # Bounds of parameter (z. B. [-1, 1])
        self.weight_bounds: tuple[float, float] | None = None
        self.bias_bounds: tuple[float, float] | None = None

        # EvoControlParams
        self.evo_params = EvoControlParams()

    def apply_config(self, cfg: ModuleConfig) -> None:

        if not isinstance(cfg, EvoNetComponentConfig):
            raise TypeError("Expected EvoNetComponentConfig")

        evo_params = self.evo_params

        # Assign dimensions
        self.dim = cfg.dim

        # Bounds
        self.weight_bounds = cfg.weight_bounds or (-1.0, 1.0)
        self.bias_bounds = cfg.bias_bounds or (-0.5, 0.5)

        # Mutation
        mutation_cfg = cfg.mutation
        if mutation_cfg is None:
            raise ValueError("Mutation config is required for ParaEvoNet.")

        evo_params.mutation_strategy = mutation_cfg.strategy

        # Strategy-specific mutation params
        if self.evo_params.mutation_strategy == MutationStrategy.CONSTANT:
            self.evo_params.mutation_probability = mutation_cfg.probability
            self.evo_params.mutation_strength = mutation_cfg.strength

        elif evo_params.mutation_strategy == MutationStrategy.EXPONENTIAL_DECAY:
            evo_params.min_mutation_probability = mutation_cfg.min_probability
            evo_params.max_mutation_probability = mutation_cfg.max_probability
            evo_params.min_mutation_strength = mutation_cfg.min_strength
            evo_params.max_mutation_strength = mutation_cfg.max_strength

        elif evo_params.mutation_strategy == MutationStrategy.ADAPTIVE_GLOBAL:
            evo_params.mutation_probability = mutation_cfg.init_probability
            evo_params.mutation_strength = mutation_cfg.init_strength
            evo_params.min_mutation_probability = mutation_cfg.min_probability
            evo_params.max_mutation_probability = mutation_cfg.max_probability
            evo_params.min_mutation_strength = mutation_cfg.min_strength
            evo_params.max_mutation_strength = mutation_cfg.max_strength
            evo_params.min_diversity_threshold = mutation_cfg.min_diversity_threshold
            evo_params.max_diversity_threshold = mutation_cfg.max_diversity_threshold
            evo_params.mutation_inc_factor = mutation_cfg.increase_factor
            evo_params.mutation_dec_factor = mutation_cfg.decrease_factor

        elif evo_params.mutation_strategy == MutationStrategy.ADAPTIVE_INDIVIDUAL:
            evo_params.min_mutation_strength = mutation_cfg.min_strength
            evo_params.max_mutation_strength = mutation_cfg.max_strength

        elif evo_params.mutation_strategy == MutationStrategy.ADAPTIVE_PER_PARAMETER:
            evo_params.min_mutation_strength = mutation_cfg.min_strength
            evo_params.max_mutation_strength = mutation_cfg.max_strength

        else:
            raise ValueError(
                f"Unknown mutation strategy: {evo_params.mutation_strategy}"
            )

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
        mutate_weights(self.net, std=self.evo_params.mutation_strength)
        mutate_biases(self.net)

    def crossover_with(self, partner: ParaBase) -> None:
        # Placeholder
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
