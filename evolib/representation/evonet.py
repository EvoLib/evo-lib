"""
EvoLib wrapper for EvoNet.

Implements the ParaBase interface for use within EvoLib's evolutionary pipeline.
Supports mutation, crossover, vector conversion, and configuration.
"""

import random as rng
from typing import Any, Optional

import numpy as np
from evonet.activation import random_function_name
from evonet.core import Nnet
from evonet.enums import NeuronRole
from evonet.mutation import mutate_activation, mutate_bias, mutate_weight

from evolib.config.base_component_config import StructuralMutationConfig
from evolib.config.evonet_component_config import EvoNetComponentConfig
from evolib.interfaces.enums import MutationStrategy
from evolib.interfaces.types import ModuleConfig
from evolib.operators.evonet_structual_mutation import mutate_structure
from evolib.operators.mutation import (
    adapt_mutation_probability_by_diversity,
    adapt_mutation_strength,
    adapt_mutation_strength_by_diversity,
    adapted_tau,
    exponential_mutation_probability,
    exponential_mutation_strength,
)
from evolib.representation._apply_config_mapping import (
    apply_crossover_config,
    apply_mutation_config,
)
from evolib.representation.base import ParaBase
from evolib.representation.evo_params import EvoControlParams


def _append_if_not_none(parts: list[str], prefix: str, value: Any) -> None:
    if value is not None:
        parts.append(f"{prefix}={value:.4f}")


class EvoNet(ParaBase):
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
        self.evo_params: EvoControlParams = EvoControlParams()
        # Optional override for biases; if None, fall back to self.evo_params
        self.bias_evo_params: Optional[EvoControlParams] = None

        # Optional override for activation mutation
        self.activation_probability: float | None = None
        self.allowed_activations: list[str] | None = None

        # Optional for structur mutation
        self.structural_cfg: StructuralMutationConfig | None = None

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
            raise ValueError("Mutation config is required for EvoNet.")

        # Global settings
        evo_params.mutation_strategy = cfg.mutation.strategy
        apply_mutation_config(self.evo_params, cfg.mutation)

        # Optional per-scope override for biases
        if cfg.mutation.biases is not None:
            self.bias_evo_params = EvoControlParams()
            apply_mutation_config(self.bias_evo_params, cfg.mutation.biases)

        # Optional activation mutatation
        if cfg.mutation.activations is not None:
            self.activation_probability = cfg.mutation.activations.probability
            self.allowed_activations = cfg.mutation.activations.allowed

        if cfg.mutation.structural is not None:
            self.structural_cfg = cfg.mutation.structural

        # Apply crossover config
        apply_crossover_config(evo_params, cfg.crossover)

        if isinstance(cfg.activation, list):
            activations = cfg.activation
        else:
            # Apply the same activation to all hidden/output layers,
            # but force input layer to be linear
            activations = ["linear"] + [cfg.activation] * (len(cfg.dim) - 1)

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

            if num_neurons > 0:
                self.net.add_neuron(
                    count=num_neurons, activation=activation_name, role=role
                )

    def calc(self, input_values: list[float]) -> list[float]:
        return self.net.calc(input_values)

    def mutate(self) -> None:

        # Weights
        if self.evo_params.mutation_strength is None:
            raise ValueError("mutation_strength must be set.")

        mutation_strength = self.evo_params.mutation_strength
        mutation_probability = self.evo_params.mutation_probability or 1.0
        low, high = self.weight_bounds or (-np.inf, np.inf)

        for connection in self.net.get_all_connections():
            if rng.random() < mutation_probability:
                mutate_weight(connection, std=mutation_strength)
                connection.weight = np.clip(connection.weight, low, high)

        # Biases (optional override)
        if self.bias_evo_params is not None:
            bias_strength = self.bias_evo_params.mutation_strength or mutation_strength
            bias_probability = (
                self.bias_evo_params.mutation_probability or mutation_probability
            )
        else:
            bias_strength, bias_probability = mutation_strength, mutation_probability

        low, high = self.bias_bounds or (-np.inf, np.inf)
        for neuron in self.net.get_all_neurons():
            if rng.random() < bias_probability and neuron.role != NeuronRole.INPUT:
                mutate_bias(neuron, std=bias_strength)
                neuron.bias = np.clip(neuron.bias, low, high)

        # Activations
        if self.activation_probability and self.activation_probability > 0.0:
            for neuron in self.net.get_all_neurons():
                if (
                    rng.random() < self.activation_probability
                    and neuron.role != NeuronRole.INPUT
                ):
                    mutate_activation(neuron, activations=self.allowed_activations)

        # Structure
        if self.structural_cfg is not None:
            mutate_structure(self.net, self.structural_cfg)

    def crossover_with(self, partner: ParaBase) -> None:
        """
        Weight-level crossover if vectors are compatible.

        No structural crossover.
        """

        if not isinstance(partner, EvoNet):
            return

        if self.evo_params._crossover_fn is None:
            return

        # Weights Crossover
        weights1 = self.get_weights()
        weights2 = partner.get_weights()
        if weights1.shape != weights2.shape:
            # Different topology or parameter count -> skip crossover
            return

        result = self.evo_params._crossover_fn(weights1, weights2)

        if isinstance(result, tuple):
            child1, child2 = result
        else:
            child1 = child2 = result

        if self.weight_bounds is None or partner.weight_bounds is None:
            raise ValueError("Both participants must define bounds before crossover.")

        min_val, max_val = self.weight_bounds
        self.set_weights(np.clip(child1, min_val, max_val))

        min_val_p, max_val_p = partner.weight_bounds
        partner.set_weights(np.clip(child2, min_val_p, max_val_p))

        # Biases Crossover
        biases1 = self.get_biases()
        biases2 = partner.get_biases()
        if biases1.shape != biases2.shape:
            # Different topology or parameter count -> skip crossover
            return

        result = self.evo_params._crossover_fn(biases1, biases2)

        if isinstance(result, tuple):
            child1, child2 = result
        else:
            child1 = child2 = result

        if self.bias_bounds is None or partner.bias_bounds is None:
            raise ValueError("Both participants must define bounds before crossover.")

        min_val, max_val = self.bias_bounds
        self.set_biases(np.clip(child1, min_val, max_val))

        min_val_p, max_val_p = partner.bias_bounds
        partner.set_biases(np.clip(child2, min_val_p, max_val_p))

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
            if ep.tau is None or ep.tau == 0.0:
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

        # If Bias-Override exists
        if self.bias_evo_params is not None:
            bep = self.bias_evo_params
            if ep.mutation_strategy == MutationStrategy.EXPONENTIAL_DECAY:
                bep.mutation_strength = exponential_mutation_strength(
                    bep, generation, max_generations
                )
                bep.mutation_probability = exponential_mutation_probability(
                    bep, generation, max_generations
                )

            elif ep.mutation_strategy == MutationStrategy.ADAPTIVE_GLOBAL:
                if diversity_ema is None:
                    raise ValueError(
                        "diversity_ema must be provided for ADAPTIVE_GLOBAL (biases)"
                    )
                if bep.mutation_strength is None or bep.mutation_probability is None:
                    raise ValueError(
                        "biases override for ADAPTIVE_GLOBAL requires both "
                        "'init_strength' and 'init_probability'."
                    )
                bep.mutation_probability = adapt_mutation_probability_by_diversity(
                    bep.mutation_probability, diversity_ema, bep
                )
                bep.mutation_strength = adapt_mutation_strength_by_diversity(
                    bep.mutation_strength, diversity_ema, bep
                )

            elif ep.mutation_strategy == MutationStrategy.ADAPTIVE_INDIVIDUAL:
                # Ensure tau is initialized
                if bep.tau is None or bep.tau == 0.0:
                    bep.tau = adapted_tau(len(self.get_vector()))

                if (
                    bep.min_mutation_strength is None
                    or bep.max_mutation_strength is None
                ):
                    raise ValueError(
                        "biases override requires min/max mutation_strength for "
                        "ADAPTIVE_INDIVIDUAL."
                    )
                if self.bias_bounds is None:
                    raise ValueError("bias_bounds must be set for bias adaptation.")
                if bep.mutation_strength is None:
                    bep.mutation_strength = np.random.uniform(
                        bep.min_mutation_strength, bep.max_mutation_strength
                    )
                bep.mutation_strength = adapt_mutation_strength(bep, self.bias_bounds)

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

        _append_if_not_none(parts, "sigma", ep.mutation_strength)
        _append_if_not_none(parts, "p", ep.mutation_probability)
        _append_if_not_none(parts, "tau", ep.tau)

        if self.bias_evo_params is not None:
            _append_if_not_none(
                parts, "sigma_bias", self.bias_evo_params.mutation_strength
            )
            _append_if_not_none(
                parts, "p_bias", self.bias_evo_params.mutation_probability
            )

        if self.activation_probability is not None:
            _append_if_not_none(parts, "p_act", self.activation_probability)

        return " | ".join(parts)

    def print_status(self) -> None:
        print(f"[EvoNet] : {self.net} ")

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
