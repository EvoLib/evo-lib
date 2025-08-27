import math

import numpy as np
import pytest
from evonet.enums import NeuronRole

from evolib.representation.evonet import EvoNet


def _build_net_2_3_1(para: EvoNet) -> None:
    """Build a small 2->3->1 feed-forward net with linear activations."""
    para.net.add_layer()  # L0
    para.net.add_layer()  # L1
    para.net.add_layer()  # L2
    para.net.add_neuron(
        layer_idx=0,
        role=NeuronRole.INPUT,
        activation="linear",
        count=2,
        connect_layer=False,
    )
    para.net.add_neuron(
        layer_idx=1,
        role=NeuronRole.HIDDEN,
        activation="linear",
        count=3,
        connect_layer=True,
    )
    para.net.add_neuron(
        layer_idx=2,
        role=NeuronRole.OUTPUT,
        activation="linear",
        count=1,
        connect_layer=True,
    )


def _build_net_1_1(para: EvoNet) -> None:
    """Build a minimal 1->1 linear net."""
    para.net.add_layer()  # L0
    para.net.add_layer()  # L1
    para.net.add_neuron(
        layer_idx=0,
        role=NeuronRole.INPUT,
        activation="linear",
        count=1,
        connect_layer=False,
    )
    para.net.add_neuron(
        layer_idx=1,
        role=NeuronRole.OUTPUT,
        activation="linear",
        count=1,
        connect_layer=True,
    )


def test_paraevo_vector_roundtrip() -> None:
    """
    Ensure EvoNet vector I/O is a bijection:

    set_vector(get_vector()) keeps parameters unchanged, and manual vectors round-trip
    verbatim.
    """
    para = EvoNet()
    _build_net_2_3_1(para)

    n_w = para.net.num_weights
    n_b = para.net.num_biases
    assert n_w > 0 and n_b > 0

    w_vec = np.linspace(-1.0, 1.0, num=n_w, dtype=float)
    b_vec = np.linspace(0.0, 0.3, num=n_b, dtype=float)
    vec = np.concatenate([w_vec, b_vec], dtype=float)

    para.set_vector(vec)
    out = para.get_vector()

    assert out.shape == vec.shape
    assert np.allclose(out, vec)
    assert np.allclose(para.net.get_weights(), w_vec)
    assert np.allclose(para.net.get_biases(), b_vec)


def test_paraevo_forward_1x1_linear() -> None:
    """
    For a 1->1 linear network, y must equal w*x + b.

    This verifies that set_vector() actually affects the forward pass.
    """
    para = EvoNet()
    _build_net_1_1(para)

    assert para.net.num_weights == 1
    assert para.net.num_biases == 1

    w, b = 2.0, 0.5
    para.set_vector(np.array([w, b], dtype=float))

    x = 3.0
    y = para.calc([x])
    assert isinstance(y, list) and len(y) == 1
    assert math.isclose(y[0], w * x + b, rel_tol=1e-9)


def test_paraevo_vector_length_matches_params() -> None:
    """get_vector() length must equal num_params (weights + biases)."""
    para = EvoNet()
    _build_net_2_3_1(para)

    vec = para.get_vector()
    assert vec.ndim == 1
    assert vec.size == para.net.num_params


def test_paraevo_set_vector_length_mismatch_raises() -> None:
    """set_vector() must raise ValueError if length does not match num_params."""
    para = EvoNet()
    _build_net_1_1(para)

    good_len = para.net.num_params
    bad_vec = np.zeros(good_len + 1, dtype=float)

    with pytest.raises(ValueError):
        para.set_vector(bad_vec)
