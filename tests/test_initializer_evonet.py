from evolib.config.schema import FullConfig
from evolib.initializers.registry import get_initializer
from evolib.representation.evonet import EvoNet


def test_normal_initializer_evonet_builds_expected_structure() -> None:
    config = FullConfig(
        parent_pool_size=1,
        offspring_pool_size=1,
        max_generations=1,
        num_elites=0,
        max_indiv_age=0,
        modules={
            "brain": {
                "type": "evonet",
                "dim": [2, 3, 1],
                "activation": "linear",
                "initializer": "normal_evonet",
                "mutation": {
                    "strategy": "constant",
                    "strength": 0.1,
                    "probability": 1.0,
                },
            }
        },
    )

    init_fn = get_initializer("normal_evonet")
    para = init_fn(config, "brain")
    assert isinstance(para, EvoNet)
    net = para.net

    # Check structure: number of layers
    assert len(net.layers) == 3

    # Count total neurons across layers
    total_neurons = sum(len(layer.neurons) for layer in net.layers)
    assert total_neurons == 6

    # Count output neurons
    output_neurons = net.layers[-1].neurons
    assert len(output_neurons) == 1

    # Check connections
    connection_count = len(net.get_all_connections())
    assert connection_count == (2 * 3) + (3 * 1)  # fully connected
