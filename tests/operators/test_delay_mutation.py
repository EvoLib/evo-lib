from __future__ import annotations

import random

from evonet.enums import ConnectionType, NeuronRole

from evolib.config.base_component_config import DelayMutationConfig
from evolib.representation.evonet import EvoNet


def _build_recurrent_evonet(delay: int = 3) -> EvoNet:
    ev = EvoNet()
    net = ev.net
    net.add_layer(3)

    inp = net.add_neuron(0, role=NeuronRole.INPUT, count=1, connection_init="none")[0]
    hid = net.add_neuron(1, role=NeuronRole.HIDDEN, count=1, connection_init="none")[0]
    out = net.add_neuron(2, role=NeuronRole.OUTPUT, count=1, connection_init="none")[0]

    net.add_connection(inp, hid, weight=1.0, conn_type=ConnectionType.STANDARD)
    net.add_connection(hid, out, weight=1.0, conn_type=ConnectionType.STANDARD)
    net.add_connection(
        hid, hid, weight=1.0, conn_type=ConnectionType.RECURRENT, delay=delay
    )

    return ev


def test_delay_mutation_delta_step_clamps() -> None:
    random.seed(1)

    ev = _build_recurrent_evonet(delay=1)

    # Disable other mutation channels if they exist
    if hasattr(ev, "evo_params"):
        ev.evo_params.mutation_strength = 0.0
        ev.evo_params.mutation_probability = 0.0

    ev.delay_mutation_cfg = DelayMutationConfig(
        probability=1.0,
        mode="delta_step",
        delta=1,
        bounds=(1, 2),
    )

    ev.mutate()

    rec = [
        c for c in ev.net.get_all_connections() if (c.type is ConnectionType.RECURRENT)
    ][0]
    assert 1 <= rec.delay <= 2


def test_delay_mutation_resample_within_bounds() -> None:
    random.seed(2)

    ev = _build_recurrent_evonet(delay=5)

    if hasattr(ev, "evo_params"):
        ev.evo_params.mutation_strength = 0.0
        ev.evo_params.mutation_probability = 0.0

    ev.delay_mutation_cfg = DelayMutationConfig(
        probability=1.0,
        mode="resample",
        delta=1,  # ignored for resample
        bounds=(1, 3),
    )

    ev.mutate()

    rec = [
        c for c in ev.net.get_all_connections() if (c.type is ConnectionType.RECURRENT)
    ][0]
    assert 1 <= rec.delay <= 3
