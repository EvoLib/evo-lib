from evolib.config.base_component_config import (
    AddConnection,
    AddNeuron,
    RemoveConnection,
    RemoveNeuron,
    StructuralMutationConfig,
    StructuralTopology,
)
from evolib.config.schema import FullConfig
from evolib.initializers.evonet_initializers import initializer_normal_evonet
from evolib.operators.evonet_structural_mutation import (
    mutate_structure,
)
from evolib.representation.evonet import EvoNet


def make_minimal_evonet() -> EvoNet:
    """Create a minimal EvoNet with valid FullConfig."""
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

    cfg = StructuralMutationConfig(
        add_connection=AddConnection(probability=1.0, max=3),
        remove_connection=None,
        add_neuron=None,
        remove_neuron=None,
        topology=StructuralTopology(),
    )

    mutate_structure(evonet.net, cfg)
    num_after = len(evonet.net.get_all_connections())

    assert num_after > num_before


def test_remove_connection() -> None:
    evonet = make_minimal_evonet()
    evonet.net.add_connection(
        evonet.net.layers[0].neurons[0], evonet.net.layers[1].neurons[0]
    )
    num_before = len(evonet.net.get_all_connections())

    cfg = StructuralMutationConfig(
        remove_connection=RemoveConnection(probability=1.0, max=3),
        add_connection=None,
        add_neuron=None,
        remove_neuron=None,
        topology=StructuralTopology(),
    )

    mutate_structure(evonet.net, cfg)
    num_after = len(evonet.net.get_all_connections())

    assert num_before > num_after


def test_add_neuron() -> None:
    evonet = make_minimal_evonet()
    num_before = len(evonet.net.get_all_neurons())

    cfg = StructuralMutationConfig(
        add_neuron=AddNeuron(probability=1.0, init_connection_ratio=1.0, init="random"),
        remove_neuron=None,
        add_connection=None,
        remove_connection=None,
        topology=StructuralTopology(),
    )

    mutate_structure(evonet.net, cfg)
    num_after = len(evonet.net.get_all_neurons())

    assert num_after > num_before


def test_remove_neuron() -> None:
    evonet = make_minimal_evonet()
    num_before = len(evonet.net.get_all_neurons())

    cfg = StructuralMutationConfig(
        remove_neuron=RemoveNeuron(probability=1.0),
        add_neuron=None,
        add_connection=None,
        remove_connection=None,
        topology=StructuralTopology(),
    )

    mutate_structure(evonet.net, cfg)
    num_after = len(evonet.net.get_all_neurons())

    assert num_after < num_before


def test_max_neurons_respected() -> None:
    evonet = make_minimal_evonet()
    non_input_count = evonet.net.num_hidden

    cfg = StructuralMutationConfig(
        add_neuron=AddNeuron(probability=1.0, init_connection_ratio=1.0),
        remove_neuron=None,
        add_connection=None,
        remove_connection=None,
        topology=StructuralTopology(max_neurons=non_input_count),
    )

    mutate_structure(evonet.net, cfg)
    assert evonet.net.num_hidden == non_input_count


def test_max_connections_respected() -> None:
    evonet = make_minimal_evonet()

    # force two connections
    evonet.net.add_connection(
        evonet.net.layers[0].neurons[0], evonet.net.layers[1].neurons[0]
    )
    evonet.net.add_connection(
        evonet.net.layers[0].neurons[0], evonet.net.layers[1].neurons[1]
    )

    num_before = len(evonet.net.get_all_connections())

    cfg = StructuralMutationConfig(
        add_connection=AddConnection(probability=1.0, max=5),
        remove_connection=None,
        add_neuron=None,
        remove_neuron=None,
        topology=StructuralTopology(max_connections=2),
    )

    mutate_structure(evonet.net, cfg)
    num_after = len(evonet.net.get_all_connections())

    assert num_before == num_after
