from evolib.config.schema import FullConfig
from evolib.initializers.registry import get_initializer


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
                "weight_bounds": (-0.9, 0.9),
                "bias_bounds": (-0.1, 0.1),
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

    # Check structure
    assert hasattr(para, "net")
    net = para.net

    assert len(net.input_neurons) == 2
    assert len(net.output_neurons) == 1
    assert len(net.neurons) == 6
    assert len(net.connections) == 2 * 3 + 3 * 1  # fully connected

    # Check that weights and biases are within expected bounds
    weights = net.get_weights()
    biases = net.get_biases()

    assert all(-0.9 <= w <= 0.9 for w in weights)
    assert all(-0.1 <= b <= 0.1 for b in biases)
