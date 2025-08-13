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
from evolib.operators.mutation import (
    adapt_mutation_probability_by_diversity,
    adapt_mutation_strength,
    adapt_mutation_strength_by_diversity,
    adapted_tau,
    exponential_mutation_probability,
    exponential_mutation_strength,
)
from evolib.representation._apply_config_mapping import (
    apply_mutation_config,
)
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
        if cfg.mutation is None:
            raise ValueError("Mutation config is required for ParaEvoNet.")

        evo_params.mutation_strategy = cfg.mutation.strategy

        # Apply mutation config
        apply_mutation_config(evo_params, cfg.mutation)

        # apply crossover config
        # apply_crossover_config(evo_params, cfg.crossover)

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

        if self.evo_params.mutation_strength is None:
            raise ValueError("mutation_strength must be set.")

        probability = self.evo_params.mutation_probability or 1.0

        mutate_weights(
            self.net,
            std=self.evo_params.mutation_strength,
            probability=probability,
        )

        mutate_biases(
            self.net,
            std=self.evo_params.mutation_strength,
            probability=probability,
        )

    def crossover_with(self, partner: ParaBase) -> None:
        # Placeholder
        # NOTE: Will be implementet in Phase 3
        pass

    def update_mutation_parameters(
        self, generation: int, max_generations: int, diversity_ema: float | None = None
    ) -> None:

        ep = self.evo_params
        """Update mutation parameters based on strategy and generation."""
        if ep.mutation_strategy == MutationStrategy.EXPONENTIAL_DECAY:
            ep.mutation_strength = exponential_mutation_strength(
                ep, generation, max_generations
            )

            ep.mutation_probability = exponential_mutation_probability(
                ep, generation, max_generations
            )

        elif ep.mutation_strategy == MutationStrategy.ADAPTIVE_GLOBAL:
            if diversity_ema is None:
                raise ValueError(
                    "diversity_ema must be provided" "for ADAPTIVE_GLOBAL strategy"
                )
            if ep.mutation_strength is None:
                raise ValueError(
                    "mutation_strength must be provided" "for ADAPTIVE_GLOBAL strategy"
                )
            if ep.mutation_probability is None:
                raise ValueError(
                    "mutation_probability must be provided"
                    "for ADAPTIVE_GLOBAL strategy"
                )

            ep.mutation_probability = adapt_mutation_probability_by_diversity(
                ep.mutation_probability, diversity_ema, ep
            )

            ep.mutation_strength = adapt_mutation_strength_by_diversity(
                ep.mutation_strength, diversity_ema, ep
            )

        elif ep.mutation_strategy == MutationStrategy.ADAPTIVE_INDIVIDUAL:
            # Ensure tau is initialized
            ep.tau = adapted_tau(len(self.get_vector()))

            if ep.min_mutation_strength is None or ep.max_mutation_strength is None:
                raise ValueError(
                    "min_mutation_strength and max_mutation_strength must be defined."
                )

            if self.weight_bounds is None:
                raise ValueError("bounds must be set")

            # Ensure mutation_strength is initialized
            if ep.mutation_strength is None:
                ep.mutation_strength = np.random.uniform(
                    ep.min_mutation_strength, ep.max_mutation_strength
                )

            # Perform adaptive update
            ep.mutation_strength = adapt_mutation_strength(ep, self.weight_bounds)

    def get_vector(self) -> np.ndarray:
        """Returns a flat vector of all weights and biases."""
        weights = self.net.get_weights()
        biases = self.net.get_biases()
        return np.concatenate([weights, biases])

    def set_vector(self, vector: np.ndarray) -> None:
        """Split a flat vector into weights and biases and apply them to the network."""
        vector = np.asarray(vector, dtype=float).ravel()
        n_weights = self.net.num_weights
        n_biases = self.net.num_biases
        if vector.size != (n_weights + n_biases):
            raise ValueError(
                f"Vector length mismatch: expected {n_weights + n_biases}, "
                f"got {vector.size}."
            )
        self.net.set_weights(vector[:n_weights])
        self.net.set_biases(vector[n_weights:])

    # Wrappers
    def get_weights(self) -> np.ndarray:
        """Return network weights in the canonical order defined by Nnet."""
        return self.net.get_weights()

    def set_weights(self, weights: np.ndarray) -> None:
        """Set network weights; length must match num_weights."""
        self.net.set_weights(weights)

    def get_biases(self) -> np.ndarray:
        """Return network biases (non-input neurons)."""
        return self.net.get_biases()

    def set_biases(self, biases: np.ndarray) -> None:
        """Set network biases; length must match num_biases."""
        self.net.set_biases(biases)

    def get_status(self) -> str:
        ep = self.evo_params
        parts = [
            f"layers={len(self.dim)}",
            f"weights={self.net.num_weights}",
            f"biases={self.net.num_biases}",
        ]
        if ep.mutation_strength is not None:
            parts.append(f"sigma={ep.mutation_strength:.4f}")
        if ep.mutation_probability is not None:
            parts.append(f"p={ep.mutation_probability:.4f}")
        if ep.tau:
            parts.append(f"tau={ep.tau:.4f}")
        return " | ".join(parts)

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
