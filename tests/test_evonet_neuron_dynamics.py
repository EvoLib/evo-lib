# SPDX-License-Identifier: MIT

import pytest

from evolib.config.evonet_component_config import EvoNetComponentConfig
from evolib.config.schema import FullConfig
from evolib.initializers.evonet_initializers import initializer_default_evonet


def test_neuron_dynamics_config_length_matches_dim() -> None:
    # neuron_dynamics length must match dim length
    with pytest.raises(
        ValueError, match="Length of 'neuron_dynamics' must match 'dim'"
    ):
        EvoNetComponentConfig(
            type="evonet",
            dim=[1, 2, 1],
            activation=["linear", "tanh", "tanh"],
            weights={
                "initializer": "normal",
                "std": 0.5,
                "bounds": [-1.0, 1.0],
            },
            bias={
                "initializer": "normal",
                "std": 0.1,
                "bounds": [-0.5, 0.5],
            },
            neuron_dynamics=[
                {"name": "standard", "params": {}},
                {"name": "standard", "params": {}},
                # missing third layer entry -> should fail
            ],
            mutation={"strategy": "constant", "strength": 0.1},
        )


def test_neuron_dynamics_applied_to_neurons() -> None:
    cfg = FullConfig(
        parent_pool_size=2,
        offspring_pool_size=2,
        max_generations=1,
        num_elites=0,
        random_seed=123,
        modules={
            "brain": {
                "type": "evonet",
                "dim": [1, 3, 1],
                "activation": ["linear", "tanh", "tanh"],
                "initializer": "default",
                "recurrent": "local",
                "connection_scope": "adjacent",
                "connection_density": 1.0,
                "weights": {
                    "initializer": "normal",
                    "std": 0.5,
                    "bounds": [-1.0, 1.0],
                },
                "bias": {
                    "initializer": "normal",
                    "std": 0.1,
                    "bounds": [-0.5, 0.5],
                },
                "neuron_dynamics": [
                    {"name": "standard", "params": {}},
                    {"name": "leaky", "params": {"alpha": 0.25}},
                    {"name": "standard", "params": {}},
                ],
                "mutation": {
                    "strategy": "constant",
                    "strength": 0.1,
                    "probability": 1.0,
                },
            }
        },
        evolution=None,
        selection=None,
        replacement=None,
        stopping=None,
        logging=None,
        parallel=None,
    )

    para = initializer_default_evonet(cfg, "brain")

    # Layer 0 (input): standard
    for n in para.net.layers[0].neurons:
        assert getattr(n, "dynamics_name", None) == "standard"
        assert getattr(n, "dynamics_params", None) == {}

    # Layer 1 (hidden): leaky + alpha
    for n in para.net.layers[1].neurons:
        assert getattr(n, "dynamics_name", None) == "leaky"
        assert getattr(n, "dynamics_params", None) == {"alpha": 0.25}

    # Layer 2 (output): standard
    for n in para.net.layers[2].neurons:
        assert getattr(n, "dynamics_name", None) == "standard"
        assert getattr(n, "dynamics_params", None) == {}
