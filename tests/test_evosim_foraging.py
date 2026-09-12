# SPDX-License-Identifier: MIT
"""Minimal behavioral tests for the EvoSim Foraging example."""

import math
from pathlib import Path

import pytest

from evosim.sims.foraging.objects import Food
from evosim.sims.foraging.simulation import ForagingSimulation

_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "10_evosim"
    / "01_foraging"
    / "simulation.yaml"
)


def _simulation() -> ForagingSimulation:
    return ForagingSimulation(_CONFIG_PATH)


def test_reset_with_same_seed_reproduces_initial_world() -> None:
    sim = _simulation()

    foragers_before = [
        (forager.x, forager.y, forager.heading) for forager in sim.foragers
    ]
    food_before = [(food.x, food.y) for food in sim.food]

    sim.reset(seed=sim.config.seed)

    foragers_after = [
        (forager.x, forager.y, forager.heading) for forager in sim.foragers
    ]
    food_after = [(food.x, food.y) for food in sim.food]

    for before_forager, after_forager in zip(
        foragers_before, foragers_after, strict=True
    ):
        assert after_forager == pytest.approx(before_forager)

    for before_food, after_food in zip(food_before, food_after, strict=True):
        assert after_food == pytest.approx(before_food)


def test_food_sensor_uses_toroidal_world_geometry() -> None:
    sim = _simulation()
    forager = sim.foragers[0]

    forager.x = 5.0
    forager.y = 100.0
    forager.heading = math.pi
    forager.energy = sim.config.forager.energy_capacity / 2.0

    sim.food = [
        Food(
            x=sim.config.world.width - 5.0,
            y=100.0,
            radius=sim.config.food.radius,
        )
    ]

    sensors = sim.sensor_layout(forager)
    observation = sim.observation(forager)

    expected_center = 1.0 - 10.0 / sensors[1].range

    assert observation[:3] == pytest.approx([0.0, expected_center, 0.0])
    assert observation[3] == pytest.approx(0.5)


def test_one_food_item_is_consumed_by_nearest_forager_only() -> None:
    sim = _simulation()
    first, second = sim.foragers[:2]
    sim.foragers = [first, second]

    first.x = 100.0
    first.y = 100.0
    second.x = 104.0
    second.y = 100.0

    first.energy = sim.config.forager.energy_capacity - 1.0
    second.energy = 10.0

    sim.food = [
        Food(
            x=101.0,
            y=100.0,
            radius=sim.config.food.radius,
        )
    ]

    sim._consume_food()

    assert sim.food == []
    assert sim.food_eaten == 1
    assert first.energy == pytest.approx(sim.config.forager.energy_capacity)
    assert second.energy == pytest.approx(10.0)


def test_reproduction_creates_independent_offspring_and_charges_parent() -> None:
    sim = _simulation()
    parent = sim.foragers[0]
    sim.foragers = [parent]

    cfg = sim.config.forager
    parent.energy = cfg.reproduction_threshold
    parent.lifetime_steps = cfg.min_reproduction_age_steps

    sim._reproduce()

    assert len(sim.foragers) == 2
    assert sim.births == 1
    assert parent.energy == pytest.approx(
        cfg.reproduction_threshold - cfg.reproduction_cost
    )

    child = sim.foragers[1]
    assert child.energy == pytest.approx(cfg.offspring_energy)
    assert child.indiv is not parent.indiv
    assert child.indiv.para is not parent.indiv.para
