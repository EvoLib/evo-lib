# SPDX-License-Identifier: MIT
"""
Tests for EvoNet deepcopy behavior.

Goal: Ensure temporal execution state (delay buffers, neuron state) does not leak
into offspring when EvoNet is deep-copied.
"""

from __future__ import annotations

import copy

from evonet.enums import ConnectionType, NeuronRole

from evolib.representation.evonet import EvoNet


def _build_minimal_recurrent_evonet(delay: int = 3) -> EvoNet:
    """
    Build a tiny EvoNet with a single recurrent connection so delay buffers exist.

    Layout:
        input(1) -> hidden(1) -> output(1)
                     ^  |
                     |__| recurrent (delay=delay)

    Notes:
        - We only need the recurrent edge to create and fill a delay history buffer.
        - We keep weights deterministic (default zeros are fine).
    """
    ev = EvoNet()
    net = ev.net

    # Create 3 layers: input, hidden, output
    net.add_layer(3)

    inp = net.add_neuron(
        layer_idx=0, role=NeuronRole.INPUT, count=1, connection_init="none"
    )[0]
    hid = net.add_neuron(
        layer_idx=1, role=NeuronRole.HIDDEN, count=1, connection_init="none"
    )[0]
    out = net.add_neuron(
        layer_idx=2, role=NeuronRole.OUTPUT, count=1, connection_init="none"
    )[0]

    net.add_connection(inp, hid, weight=1.0, conn_type=ConnectionType.STANDARD)
    net.add_connection(hid, out, weight=1.0, conn_type=ConnectionType.STANDARD)

    # Recurrent edge with delay buffer
    net.add_connection(
        hid, hid, weight=1.0, conn_type=ConnectionType.RECURRENT, delay=delay
    )

    return ev


def test_evonet_deepcopy_clears_delay_buffers() -> None:
    ev = _build_minimal_recurrent_evonet(delay=3)

    # Fill temporal state (push into recurrent history buffers)
    ev.net.calc([1.0])
    ev.net.calc([0.0])

    conns = ev.net.get_all_connections()
    recurrent = [c for c in conns if c.type is ConnectionType.RECURRENT]
    assert recurrent, "Test requires at least one recurrent connection"

    # Verify history got filled
    assert any(c._history is not None and len(c._history) > 0 for c in recurrent)

    # Deepcopy must not inherit history
    ev2 = copy.deepcopy(ev)

    conns2 = ev2.net.get_all_connections()
    recurrent2 = [c for c in conns2 if c.type is ConnectionType.RECURRENT]
    assert recurrent2, "Copied net must still have recurrent connections"

    assert all(c._history is None or len(c._history) == 0 for c in recurrent2)
