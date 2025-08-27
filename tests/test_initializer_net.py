from typing import cast

import numpy as np

from evolib.config.schema import FullConfig
from evolib.initializers.registry import get_initializer
from evolib.representation.netvector import NetVector
from evolib.representation.vector import Vector


def test_normal_initializer_netvector_builds_expected_structure() -> None:
    layer_dims = [2, 3, 1]
    activation = "linear"

    expected_net = NetVector(dim=layer_dims, activation=activation)
    expected_param_count = expected_net.n_parameters

    config = FullConfig(
        parent_pool_size=1,
        offspring_pool_size=1,
        max_generations=1,
        num_elites=0,
        max_indiv_age=0,
        modules={
            "brain": {
                "type": "vector",
                "structure": "net",
                "dim": layer_dims,
                "activation": activation,
                "initializer": "normal_vector",
                "bounds": (-1.0, 1.0),
                "init_bounds": (-1.0, 1.0),
                "mean": 0.0,
                "std": 0.5,
                "mutation": {
                    "strategy": "constant",
                    "strength": 0.1,
                    "probability": 1.0,
                },
            }
        },
    )

    init_fn = get_initializer("normal_net")
    para = cast(Vector, init_fn(config, "brain"))

    # Check shape and size
    assert isinstance(para.vector, np.ndarray)
    assert para.vector.shape == (expected_param_count,)
    assert para.dim == expected_param_count

    # Check value range based on configured bounds
    low, high = config.modules["brain"].bounds
    assert np.all(para.vector >= low)
    assert np.all(para.vector <= high)

    # Reconstruct structure using NetVector logic
    weights, biases = expected_net._unpack_parameters(para.vector)

    assert len(weights) == len(layer_dims) - 1
    assert len(biases) == len(layer_dims) - 1

    for i in range(len(weights)):
        assert weights[i].shape == (layer_dims[i + 1], layer_dims[i])
        assert biases[i].shape == (layer_dims[i + 1],)

    # Optional: forward pass sanity check
    x = np.random.randn(layer_dims[0])
    y = expected_net.forward(x, para.vector)
    assert y.shape == (layer_dims[-1],)
