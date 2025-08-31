from evolib.config.base_component_config import StructuralMutationConfig
from evolib.config.schema import FullConfig
from evolib.initializers.evonet_initializers import initializer_normal_evonet
from evolib.operators.evonet_structual_mutation import (
    count_non_input_neurons,
    mutate_structure,
)
from evolib.representation.evonet import EvoNet


def make_minimal_evonet() -> EvoNet:
    config_dict = {
        "parent_pool_size": 10,
        "offspring_pool_size": 20,
        "max_generations": 5,
        "max_indiv_age": 0,
        "num_elites": 1,
        "evolution": {"strategy": "mu_plus_lambda"},
        "modules": {
            "brain": {
                "type": "evonet",
                "dim": [2, 3, 1],
                "activation": ["linear", "tanh", "sigmoid"],
                "initializer": "normal_evonet",
                "weight_bounds": [-1.0, 1.0],
                "bias_bounds": [-0.5, 0.5],
                "mutation": {
                    "strategy": "constant",
                    "strength": 0.1,
                    "probability": 1.0,
                },
            }
        },
    }
    full_config = FullConfig.model_validate(config_dict)
    return initializer_normal_evonet(full_config, module="brain")


def test_add_connection() -> None:
    evonet = make_minimal_evonet()
    num_before = len(evonet.net.get_all_connections())
    cfg = StructuralMutationConfig(add_connection=1.0)
    mutate_structure(evonet.net, cfg)
    num_after = len(evonet.net.get_all_connections())
    assert num_after > num_before


def test_remove_connection() -> None:
    evonet = make_minimal_evonet()
    evonet.net.add_connection(
        evonet.net.layers[0].neurons[0], evonet.net.layers[1].neurons[0]
    )
    num_before = len(evonet.net.get_all_connections())
    cfg = StructuralMutationConfig(remove_connection=1.0)
    mutate_structure(evonet.net, cfg)
    num_after = len(evonet.net.get_all_connections())
    assert num_before > num_after


def test_add_neuron() -> None:
    evonet = make_minimal_evonet()
    num_before = len(evonet.net.get_all_neurons())
    cfg = StructuralMutationConfig(add_neuron=1.0)
    mutate_structure(evonet.net, cfg)
    num_after = len(evonet.net.get_all_neurons())
    assert num_after > num_before


def test_remove_neuron() -> None:
    evonet = make_minimal_evonet()
    cfg = StructuralMutationConfig(remove_neuron=1.0)
    num_before = len(evonet.net.get_all_neurons())
    mutate_structure(evonet.net, cfg)
    num_after = len(evonet.net.get_all_neurons())
    assert num_before > num_after


def test_split_connection() -> None:
    evonet = make_minimal_evonet()
    evonet.net.add_connection(
        evonet.net.layers[0].neurons[0], evonet.net.layers[1].neurons[0]
    )
    n_num_before = len(evonet.net.get_all_neurons())
    c_num_before = len(evonet.net.get_all_connections())

    cfg = StructuralMutationConfig(split_connection=1.0)
    mutate_structure(evonet.net, cfg)

    n_num_after = len(evonet.net.get_all_neurons())
    c_num_after = len(evonet.net.get_all_connections())

    assert n_num_after > n_num_before and c_num_after > c_num_before


def test_max_nodes_respected() -> None:
    evonet = make_minimal_evonet()
    non_input_count = count_non_input_neurons(evonet.net)
    cfg = StructuralMutationConfig(add_neuron=1.0, max_nodes=non_input_count)
    mutate_structure(evonet.net, cfg)
    assert count_non_input_neurons(evonet.net) == non_input_count


def test_max_edges_respected() -> None:
    evonet = make_minimal_evonet()
    evonet.net.add_connection(
        evonet.net.layers[0].neurons[0], evonet.net.layers[1].neurons[0]
    )
    evonet.net.add_connection(
        evonet.net.layers[0].neurons[0], evonet.net.layers[1].neurons[1]
    )
    num_before = len(evonet.net.get_all_connections())
    cfg = StructuralMutationConfig(add_connection=1.0, max_edges=2)
    mutate_structure(evonet.net, cfg)
    num_after = len(evonet.net.get_all_connections())
    assert num_before == num_after
