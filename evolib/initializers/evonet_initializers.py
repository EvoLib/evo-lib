# SPDX-License-Identifier: MIT
"""
Initializers for EvoNet networks.

These initializers convert a module configuration with `type: evonet` into a fully
initialized EvoNet instance.
"""

import numpy as np
from evonet.activation import random_function_name
from evonet.enums import ConnectionType, NeuronRole

from evolib.config.evonet_component_config import EvoNetComponentConfig
from evolib.config.schema import FullConfig
from evolib.interfaces.enum_helpers import resolve_recurrent_kinds
from evolib.representation.evonet import EvoNet


def _build_architecture(para: EvoNet, cfg: EvoNetComponentConfig) -> None:
    """
    Build the EvoNet architecture (layers, neurons, activations) from config.

    Args:
        para (EvoNet): The EvoNet instance (already has parameters set).
        cfg (EvoNetComponentConfig): Config with architecture definition.
    """
    # Activation functions per layer
    if isinstance(cfg.activation, list):
        activations = cfg.activation
    else:
        # Input layer linear, others same activation
        activations = ["linear"] + [cfg.activation] * (len(cfg.dim) - 1)

    for layer_idx, num_neurons in enumerate(cfg.dim):
        if num_neurons == 0:
            continue

        activation_name = activations[layer_idx]
        if activation_name == "random":
            if cfg.activations_allowed is not None:
                activation_name = random_function_name(cfg.activations_allowed)
            else:
                activation_name = random_function_name()

        para.net.add_layer()

        if layer_idx == 0:
            role = NeuronRole.INPUT
        elif layer_idx == len(cfg.dim) - 1:
            role = NeuronRole.OUTPUT
        else:
            role = NeuronRole.HIDDEN

        recurrent_kinds = resolve_recurrent_kinds(cfg.recurrent)
        para.net.add_neuron(
            count=num_neurons,
            activation=activation_name,
            role=role,
            connection_init="zero",
            bias=0.0,
            recurrent=recurrent_kinds if role != NeuronRole.INPUT else None,
        )


def initializer_normal_evonet(config: FullConfig, module: str) -> EvoNet:
    """
    Initializes an EvoNet with normally distributed weights.

    Args:
        config (FullConfig): Full experiment configuration
        module (str): Module name (e.g. "brain")

    Returns:
        EvoNet: Initialized EvoNet representation
    """
    para = EvoNet()
    cfg = config.modules[module].model_copy(deep=True)
    para.apply_config(cfg)

    _build_architecture(para, cfg)

    # Initialize weights and biases with normal distribution
    weights = np.random.normal(loc=0.0, scale=0.5, size=para.net.num_weights)
    biases = np.random.normal(loc=0.0, scale=0.5, size=para.net.num_biases)

    para.net.set_weights(weights)
    para.net.set_biases(biases)
    return para


def initializer_random_evonet(config: FullConfig, module: str) -> EvoNet:
    """
    Initializes an EvoNet with uniformly random weights and biases within init_bounds
    (fallback: weight_bounds, bias_bounds).

    Args:
        config (FullConfig): Full experiment configuration
        module (str): Module name

    Returns:
        EvoNet: Initialized EvoNet representation
    """
    para = EvoNet()
    cfg = config.modules[module].model_copy(deep=True)
    para.apply_config(cfg)

    _build_architecture(para, cfg)

    # Use weight_bounds if available, else fall back to bounds
    weight_bounds = cfg.weight_bounds or (-1.0, 1.0)
    min_weight = weight_bounds[0]
    max_weight = weight_bounds[1]
    bias_bounds = cfg.bias_bounds or (-0.5, 0.5)
    min_bias = bias_bounds[0]
    max_bias = bias_bounds[1]

    weights = np.random.uniform(min_weight, max_weight, size=para.net.num_weights)
    biases = np.random.uniform(min_bias, max_bias, size=para.net.num_biases)

    para.net.set_weights(weights)
    para.net.set_biases(biases)
    return para


def initializer_zero_evonet(config: FullConfig, module: str) -> EvoNet:
    """
    Initializes an EvoNet with all weights and biases set to zero.

    Args:
        config (FullConfig): Full experiment configuration
        module (str): Module name

    Returns:
        EvoNet: Initialized EvoNet representation
    """
    para = EvoNet()
    cfg = config.modules[module].model_copy(deep=True)
    para.apply_config(cfg)

    _build_architecture(para, cfg)

    para.net.set_weights(np.zeros(para.net.num_weights))
    para.net.set_biases(np.zeros(para.net.num_biases))
    return para


def initializer_identity_evonet(config: FullConfig, module: str) -> EvoNet:
    """
    Initialize EvoNet with damped self-recurrence and zeroed weights elsewhere.

    - All feedforward weights are near-zero
    - Self-recurrent connections get weight ~0.8 (memory)
    - Biases are randomized slightly to break symmetry

    This initializer encourages stable internal state retention from the start,
    making recurrent behavior immediately available to evolution.

    Args:
        config (FullConfig): Full experiment configuration
        module (str): Name of the EvoNet module in the config

    Returns:
        EvoNet: Initialized network with identity-style dynamics
    """

    SELF_LOOP_WEIGHT = 0.8
    ALPHA = 0.01

    para = EvoNet()
    cfg = config.modules[module].model_copy(deep=True)
    para.apply_config(cfg)

    _build_architecture(para, cfg)

    para.net.set_weights(np.zeros(para.net.num_weights))
    para.net.set_biases(np.zeros(para.net.num_biases))

    for neuron in para.net.get_all_neurons():
        # Small random bias to break symmetry
        neuron.bias = np.random.uniform(-ALPHA, ALPHA)
        for connection in neuron.outgoing:
            # Damped self-recurrence: acts like memory cell
            if (
                connection.type == ConnectionType.RECURRENT
                and connection.source.id == connection.target.id
            ):
                connection.weight = SELF_LOOP_WEIGHT

            # Small random feedforward weight to allow weak stimulus flow
            if connection.type == ConnectionType.STANDARD:
                connection.weight = np.random.uniform(-ALPHA, ALPHA)

    return para
