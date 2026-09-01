from evonet import ConnectivityConfig

from evolib import EvoNet, EvoNetComponentConfig


def _make_config() -> EvoNetComponentConfig:
    return EvoNetComponentConfig.model_validate(
        {
            "type": "evonet",
            "dim": [2, 3, 1],
            "activation": "linear",
            "connectivity": {
                "scope": "adjacent",
                "density": 1.0,
            },
            "weights": {
                "initializer": "zero",
                "bounds": [-1.0, 1.0],
            },
            "bias": {
                "initializer": "zero",
                "bounds": [-0.5, 0.5],
            },
            "mutation": {
                "strategy": "constant",
                "strength": 0.1,
                "probability": 1.0,
            },
        }
    )


def test_evonet_component_config_uses_native_evonet_config() -> None:
    config = _make_config()

    assert isinstance(config.connectivity, ConnectivityConfig)


def test_evonet_from_config_does_not_require_full_config() -> None:
    config = _make_config()

    para = EvoNet.from_config(config)

    assert len(para.net.layers) == 3
    assert sum(len(layer.neurons) for layer in para.net.layers) == 6
    assert len(para.net.get_all_connections()) == 9
    assert para.weight_bounds == (-1.0, 1.0)
    assert para.bias_bounds == (-0.5, 0.5)
