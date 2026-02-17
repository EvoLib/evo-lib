# SPDX-License-Identifier: MIT
"""
Initializers for EvoNet networks.

These initializers convert a module configuration with `type: evonet` into a fully
initialized EvoNet instance.
"""

from typing import Literal

import numpy as np
from evonet.activation import random_function_name
from evonet.enums import ConnectionType, NeuronRole

from evolib.config.evonet_component_config import DelayConfig, EvoNetComponentConfig
from evolib.config.schema import FullConfig
from evolib.interfaces.enum_helpers import resolve_recurrent_kinds
from evolib.representation.evonet import EvoNet


def _clip(x: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    lo, hi = bounds
    return np.clip(x, lo, hi)


def _apply_weights_init(para: EvoNet, cfg: EvoNetComponentConfig) -> None:
    weights_cfg = cfg.weights
    size = para.net.num_weights

    if weights_cfg.initializer is None:
        return

    elif weights_cfg.initializer == "zero":
        weights = np.zeros(size, dtype=float)

    elif weights_cfg.initializer == "uniform":
        # uniform uses init_bounds if present, otherwise bounds
        lo, hi = weights_cfg.init_bounds or weights_cfg.bounds
        weights = np.random.uniform(lo, hi, size=size)

    elif weights_cfg.initializer == "normal":
        assert weights_cfg.std is not None
        weights = np.random.normal(loc=0.0, scale=weights_cfg.std, size=size)

        # clip ONLY if init_bounds explicitly provided
        if weights_cfg.init_bounds is not None:
            lo, hi = weights_cfg.init_bounds
            weights = np.clip(weights, lo, hi)

    else:
        raise ValueError(f"Unknown weights initializer: {weights_cfg.initializer}")

    para.net.set_weights(weights)


def _apply_bias_init(para: EvoNet, cfg: EvoNetComponentConfig) -> None:
    bias_cfg = cfg.bias
    size = para.net.num_biases

    if bias_cfg.initializer is None:
        return

    elif bias_cfg.initializer == "zero":
        bias = np.zeros(size, dtype=float)

    elif bias_cfg.initializer == "fixed":
        assert bias_cfg.value is not None
        bias = np.full(size, float(bias_cfg.value), dtype=float)

    elif bias_cfg.initializer == "uniform":
        lo, hi = bias_cfg.init_bounds or bias_cfg.bounds
        bias = np.random.uniform(lo, hi, size=size)

    elif bias_cfg.initializer == "normal":
        assert bias_cfg.std is not None
        bias = np.random.normal(loc=0.0, scale=bias_cfg.std, size=size)

        # clip only if init_bounds explicitly provided
        if bias_cfg.init_bounds is not None:
            lo, hi = bias_cfg.init_bounds
            bias = np.clip(bias, lo, hi)

    else:
        raise ValueError(f"Unknown bias initializer: {bias_cfg.initializer}")

    para.net.set_biases(bias)


def _apply_delay_init(para: EvoNet, cfg: EvoNetComponentConfig) -> None:
    """Initialize delay on recurrent connections only."""

    if cfg.delay is None:
        return

    delay_cfg: DelayConfig = cfg.delay

    for connection in para.net.get_all_connections():
        if connection.type is not ConnectionType.RECURRENT:
            continue

        if delay_cfg.initializer == "uniform" and delay_cfg.bounds is not None:
            assert delay_cfg.bounds is not None
            lo, hi = delay_cfg.bounds
            delay = int(np.random.randint(lo, hi + 1))
        else:
            assert delay_cfg.value is not None
            delay = int(delay_cfg.value)

        connection.set_delay(delay)


def _build_architecture(
    para: EvoNet,
    cfg: EvoNetComponentConfig,
    connection_init: Literal["random", "zero", "near_zero", "none"] = "zero",
) -> None:
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

        para.net.add_layer()

        if num_neurons == 0:
            continue

        activation_name = activations[layer_idx]
        if activation_name == "random":
            if cfg.activations_allowed is not None:
                activation_name = random_function_name(cfg.activations_allowed)
            else:
                activation_name = random_function_name()

        if layer_idx == 0:
            role = NeuronRole.INPUT
        elif layer_idx == len(cfg.dim) - 1:
            role = NeuronRole.OUTPUT
        else:
            role = NeuronRole.HIDDEN

        # resolve dynamics per layer
        if cfg.neuron_dynamics is None:
            dynamics_name = "standard"
            dynamics_params = {}
        else:
            dynamics_cfg = cfg.neuron_dynamics[layer_idx]
            dynamics_name = dynamics_cfg.name
            dynamics_params = dynamics_cfg.params or {}

        recurrent_kinds = resolve_recurrent_kinds(cfg.recurrent)
        para.net.add_neuron(
            count=num_neurons,
            activation=activation_name,
            role=role,
            connection_init=connection_init,
            bias=0.0,
            recurrent=recurrent_kinds if role != NeuronRole.INPUT else None,
            connection_scope=para.connection_scope,
            connection_density=para.connection_density,
            dynamics_name=dynamics_name,
            dynamics_params=dynamics_params,
        )


def initializer_unconnected_evonet(config: FullConfig, module: str) -> EvoNet:
    """
    Initializes an EvoNet without connections.

    Args:
        config (FullConfig): Full experiment configuration
        module (str): Module name (e.g. "brain")

    Returns:
        EvoNet: Initialized EvoNet representation
    """
    para = EvoNet()
    cfg = config.modules[module].model_copy(deep=True)
    para.apply_config(cfg)

    _build_architecture(para, cfg, connection_init="none")
    _apply_bias_init(para, cfg)

    return para


def initializer_normal_evonet(config: FullConfig, module: str) -> EvoNet:
    """
    Build a standard EvoNet architecture and initialize parameters according to the
    explicit configuration blocks.

    - Topology is created via `_build_architecture(...)`.
    - Weights are initialized using `cfg.weights`.
    - Biases are initialized using `cfg.bias`.
    - Delay (if configured) is initialized using `cfg.delay`.

    No implicit parameter initialization is performed here.
    All parameter distributions are controlled explicitly via the config.
    """

    para = EvoNet()
    cfg = config.modules[module].model_copy(deep=True)
    para.apply_config(cfg)

    _build_architecture(para, cfg, connection_init="zero")
    _apply_delay_init(para, cfg)
    _apply_weights_init(para, cfg)
    _apply_bias_init(para, cfg)
    return para


def initializer_random_evonet(config: FullConfig, module: str) -> EvoNet:
    """
    Backward-compatible alias for the standard EvoNet initializer. Will be removed.

    Parameter initialization is controlled by `cfg.weights`, `cfg.bias`,
    and `cfg.delay`.
    """
    return initializer_normal_evonet(config, module)


def initializer_zero_evonet(config: FullConfig, module: str) -> EvoNet:
    """
    Build a standard EvoNet architecture and initialize all parameters to zero.

    - All connection weights are set to 0.
    - All biases are set to 0.
    - Delay initialization follows `cfg.delay` if applicable.

    This initializer ignores `cfg.weights` and `cfg.bias` distributions.
    """

    para = EvoNet()
    cfg = config.modules[module].model_copy(deep=True)
    para.apply_config(cfg)

    _build_architecture(para, cfg, connection_init="zero")
    _apply_delay_init(para, cfg)

    para.net.set_weights(np.zeros(para.net.num_weights))
    para.net.set_biases(np.zeros(para.net.num_biases))

    return para


def initializer_identity_evonet(config: FullConfig, module: str) -> EvoNet:
    """
    Build a feedforward EvoNet and initialize parameters with an identity-like
    structure.

    - Feedforward connections are created normally.
    - Self-recurrent connections are added explicitly.
    - Weights and biases are set to fixed values
      to approximate identity behavior.

    This preset intentionally overrides standard parameter initialization.
    """

    SELF_LOOP_WEIGHT = 0.8
    ALPHA = 0.01

    para = EvoNet()
    cfg = config.modules[module].model_copy(deep=True)
    para.apply_config(cfg)

    _build_architecture(para, cfg)
    _apply_delay_init(para, cfg)

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
