# SPDX-License-Identifier: MIT
"""Minimal behavioral tests for the EvoSim Foraging example."""

import math
from pathlib import Path

import pytest

from evosim.sims.foraging.objects import Food, Poison
from evosim.sims.foraging.simulation import ForagingSimulation

_EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "10_evosim" / "01_foraging"
)
_CONFIG_PATH = _EXAMPLE_PATH / "simulation.yaml"
_POISON_CONFIG_PATH = _EXAMPLE_PATH / "simulation_poison.yaml"


def _simulation() -> ForagingSimulation:
    return ForagingSimulation(_CONFIG_PATH)


def _poison_simulation() -> ForagingSimulation:
    return ForagingSimulation(_POISON_CONFIG_PATH)


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

    assert len(observation) == 5
    assert observation[:3] == pytest.approx([0.0, expected_center, 0.0])
    assert observation[3] == pytest.approx(0.5)
    assert observation[4] == pytest.approx(0.0)


def test_feeding_cooldown_is_normalized_in_observation() -> None:
    sim = _simulation()
    forager = sim.foragers[0]
    forager.feeding_cooldown = sim.config.forager.feeding_cooldown_steps // 2

    observation = sim.observation(forager)

    assert observation[-1] == pytest.approx(0.5)


def test_poison_adds_a_second_channel_per_sensor() -> None:
    sim = _poison_simulation()
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
    sim.poison = [
        Poison(
            x=sim.config.world.width - 15.0,
            y=100.0,
            radius=sim.config.poison.radius,
        )
    ]

    sensors = sim.sensor_layout(forager)
    observation = sim.observation(forager)

    expected_food = 1.0 - 10.0 / sensors[1].range
    expected_poison = 1.0 - 20.0 / sensors[1].range

    assert len(observation) == 8
    assert observation[:6] == pytest.approx(
        [0.0, 0.0, expected_food, expected_poison, 0.0, 0.0]
    )
    assert observation[6] == pytest.approx(0.5)
    assert observation[7] == pytest.approx(0.0)


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
    assert first.feeding_cooldown == sim.config.forager.feeding_cooldown_steps
    assert second.energy == pytest.approx(10.0)


def test_feeding_cooldown_limits_food_consumption() -> None:
    sim = _simulation()
    forager = sim.foragers[0]
    sim.foragers = [forager]
    forager.x = 100.0
    forager.y = 100.0
    forager.energy = 10.0
    sim.food = [
        Food(x=100.0, y=100.0, radius=sim.config.food.radius),
        Food(x=100.0, y=100.0, radius=sim.config.food.radius),
    ]

    sim._consume_food()

    assert len(sim.food) == 1
    assert sim.food_eaten == 1
    assert forager.feeding_cooldown == sim.config.forager.feeding_cooldown_steps

    energy_after_first_food = forager.energy
    sim._consume_food()

    assert len(sim.food) == 1
    assert sim.food_eaten == 1
    assert forager.energy == pytest.approx(energy_after_first_food)


def test_one_poison_item_is_consumed_by_nearest_forager_only() -> None:
    sim = _poison_simulation()
    first, second = sim.foragers[:2]
    sim.foragers = [first, second]

    first.x = 100.0
    first.y = 100.0
    second.x = 104.0
    second.y = 100.0
    first.energy = 50.0
    second.energy = 50.0
    sim.poison = [
        Poison(
            x=101.0,
            y=100.0,
            radius=sim.config.poison.radius,
        )
    ]

    sim._consume_poison()

    assert sim.poison == []
    assert sim.poison_eaten == 1
    assert first.energy == pytest.approx(50.0 - sim.config.poison.damage)
    assert second.energy == pytest.approx(50.0)


def test_lethal_poison_removes_forager_in_same_step() -> None:
    sim = _poison_simulation()
    forager = sim.foragers[0]
    sim.foragers = [forager]
    sim.food = []

    forager.energy = sim.config.poison.damage / 2.0
    forager.age_steps = sim.config.forager.min_reproduction_age_steps
    sim.poison = [
        Poison(
            x=forager.x,
            y=forager.y,
            radius=sim.config.poison.radius,
        )
    ]

    sim.step()

    assert sim.foragers == []
    assert sim.deaths == 1
    assert sim.births == 0


def test_reproduction_creates_independent_offspring_and_charges_parent() -> None:
    sim = _simulation()
    parent = sim.foragers[0]
    sim.foragers = [parent]

    cfg = sim.config.forager
    parent.energy = cfg.reproduction_threshold
    parent.age_steps = cfg.min_reproduction_age_steps

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
