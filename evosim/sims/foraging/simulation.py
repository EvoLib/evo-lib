# SPDX-License-Identifier: MIT
"""Persistent two-dimensional Foraging simulation."""

import math
import random

import numpy as np

from evolib import EvoNet, Indiv, Vector
from evolib.representation.composite import ParaComposite
from evosim.core.geometry import (
    angle_delta,
    toroidal_displacement,
    toroidal_distance_squared,
    wrap_coordinate,
)
from evosim.core.simulation import Simulation
from evosim.sims.foraging.config import ForagingConfig
from evosim.sims.foraging.objects import Food, FoodSensor, Forager, ForagerAction


class ForagingSimulation(Simulation):
    """Persistent world with resources and synchronously updated Foragers."""

    def __init__(self, config: ForagingConfig) -> None:
        super().__init__()
        self.config = config
        self.rng = np.random.default_rng()

        self.foragers: list[Forager] = []
        self.food: list[Food] = []

        self.births = 0
        self.deaths = 0
        self.food_eaten = 0

        self.reset(seed=config.seed)

    @property
    def world_size(self) -> tuple[int, int]:
        """Return world dimensions in simulation units."""
        return self.config.world.width, self.config.world.height

    @property
    def population_size(self) -> int:
        """Return the number of currently living Foragers."""
        return len(self.foragers)

    @property
    def is_extinct(self) -> bool:
        """Return whether no living Foragers remain."""
        return not self.foragers

    @property
    def mean_energy(self) -> float:
        """Return mean energy of living Foragers, or zero after extinction."""
        if not self.foragers:
            return 0.0
        return float(np.mean([forager.energy for forager in self.foragers]))

    @property
    def oldest_lifetime_steps(self) -> int:
        """Return age of the oldest living Forager in simulation steps."""
        return max((forager.lifetime_steps for forager in self.foragers), default=0)

    def reset(self, *, seed: int | None = None) -> None:
        """Reset world, population, resources, counters, and random state."""
        if seed is None:
            seed = self.config.seed

        self.rng = np.random.default_rng(seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.step_count = 0
        self.births = 0
        self.deaths = 0
        self.food_eaten = 0

        self.foragers = [
            self._make_founder() for _ in range(self.config.world.initial_population)
        ]
        self.food = [self._make_food() for _ in range(self.config.food.initial_count)]

    def step(self) -> None:
        """Advance one synchronous simulation step."""
        actions = [self._calculate_action(forager) for forager in self.foragers]

        self._apply_actions(actions)
        self._remove_dead_foragers()
        self._consume_food()
        self._reproduce()
        self._spawn_food()

        self.step_count += 1

    def metrics(self) -> dict[str, int | float]:
        """Return the current simulation metrics without performing I/O."""
        return {
            "step": self.step_count,
            "population": self.population_size,
            "food": len(self.food),
            "births": self.births,
            "deaths": self.deaths,
            "food_eaten": self.food_eaten,
            "mean_energy": self.mean_energy,
            "oldest_lifetime": self.oldest_lifetime_steps,
        }

    def sensor_layout(self, forager: Forager) -> list[FoodSensor]:
        """Decode the local food sensors carried by one Forager."""
        angles, fovs, ranges = self._sensor_arrays(forager)

        sensors: list[FoodSensor] = []

        for angle, fov, sensor_range in zip(angles, fovs, ranges, strict=True):
            sensor = FoodSensor(
                angle=float(angle),
                fov=float(fov),
                range=float(sensor_range),
            )
            sensors.append(sensor)

        return sensors

    def observation(self, forager: Forager) -> list[float]:
        """Return local food sensor activations plus normalized energy."""
        angles, fovs, ranges = self._sensor_arrays(forager)
        sensor_values = [0.0 for _ in range(len(angles))]

        for resource in self.food:
            dx, dy = toroidal_displacement(
                forager.x,
                forager.y,
                resource.x,
                resource.y,
                self.config.world.width,
                self.config.world.height,
            )
            distance = math.hypot(dx, dy)

            if distance <= 1e-12:
                sensor_values = [1.0 for _ in sensor_values]
                continue

            bearing = math.atan2(dy, dx)
            relative_bearing = angle_delta(forager.heading, bearing)

            for index, (angle, fov, sensor_range) in enumerate(
                zip(angles, fovs, ranges, strict=True)
            ):
                if distance > sensor_range:
                    continue
                if abs(angle_delta(float(angle), relative_bearing)) > fov / 2.0:
                    continue

                strength = 1.0 - distance / sensor_range
                sensor_values[index] = max(sensor_values[index], float(strength))

        cfg = self.config.forager
        energy = forager.energy / cfg.energy_capacity
        return [*sensor_values, energy]

    def _make_founder(self) -> Forager:
        modules = self.config.modules
        para = ParaComposite(
            {
                "controller": EvoNet.from_config(modules.controller),
                "sensor_angles": Vector.from_config(modules.sensor_angles),
                "sensor_fovs": Vector.from_config(modules.sensor_fovs),
                "sensor_ranges": Vector.from_config(modules.sensor_ranges),
            }
        )
        return self._place_forager(Indiv(para), self.config.forager.initial_energy)

    def _make_offspring(self, parent: Forager) -> Forager:
        child_indiv = parent.indiv.copy(
            reset_fitness=True,
            reset_age=True,
            reset_origin=True,
        )
        child_indiv.mutate()

        spawn_angle = float(self.rng.uniform(0.0, math.tau))
        spawn_distance = parent.radius * 2.5
        x = wrap_coordinate(
            parent.x + math.cos(spawn_angle) * spawn_distance,
            self.config.world.width,
        )
        y = wrap_coordinate(
            parent.y + math.sin(spawn_angle) * spawn_distance,
            self.config.world.height,
        )

        return Forager(
            indiv=child_indiv,
            x=x,
            y=y,
            heading=float(self.rng.uniform(0.0, math.tau)),
            energy=self.config.forager.offspring_energy,
            radius=self.config.forager.radius,
        )

    def _place_forager(self, indiv: Indiv, energy: float) -> Forager:
        return Forager(
            indiv=indiv,
            x=float(self.rng.uniform(0.0, self.config.world.width)),
            y=float(self.rng.uniform(0.0, self.config.world.height)),
            heading=float(self.rng.uniform(0.0, math.tau)),
            energy=energy,
            radius=self.config.forager.radius,
        )

    def _make_food(self) -> Food:
        return Food(
            x=float(self.rng.uniform(0.0, self.config.world.width)),
            y=float(self.rng.uniform(0.0, self.config.world.height)),
            radius=self.config.food.radius,
        )

    @staticmethod
    def _composite(forager: Forager) -> ParaComposite:
        para = forager.indiv.para
        if not isinstance(para, ParaComposite):
            raise TypeError(
                "Foraging individuals require a ParaComposite representation."
            )
        return para

    def _controller(self, forager: Forager) -> EvoNet:
        controller = self._composite(forager)["controller"]
        if not isinstance(controller, EvoNet):
            raise TypeError("Foraging requires an EvoNet 'controller' component.")
        return controller

    def _sensor_vector(self, forager: Forager, name: str) -> Vector:
        vector = self._composite(forager)[name]
        if not isinstance(vector, Vector):
            raise TypeError(f"Foraging requires a Vector '{name}' component.")
        return vector

    def _sensor_arrays(
        self, forager: Forager
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return aligned angle, FOV, and range arrays for one Forager."""
        angles = self._sensor_vector(forager, "sensor_angles").vector
        fovs = self._sensor_vector(forager, "sensor_fovs").vector
        ranges = self._sensor_vector(forager, "sensor_ranges").vector
        if not (len(angles) == len(fovs) == len(ranges)):
            raise ValueError("Foraging sensor vectors must have equal size.")
        return angles, fovs, ranges

    def _calculate_action(self, forager: Forager) -> ForagerAction:
        outputs = self._controller(forager).calc(self.observation(forager))
        if len(outputs) != 2:
            raise ValueError("Foraging controller must return exactly two outputs.")

        turn = float(np.clip(outputs[0], -1.0, 1.0))
        throttle = float(np.clip(outputs[1], -1.0, 1.0))
        return ForagerAction(turn=turn, throttle=throttle)

    def _apply_actions(self, actions: list[ForagerAction]) -> None:
        cfg = self.config.forager

        for forager, action in zip(self.foragers, actions, strict=True):
            forager.heading = (
                forager.heading + action.turn * cfg.max_turn_rate
            ) % math.tau

            if action.throttle >= 0.0:
                distance = action.throttle * cfg.max_speed
            else:
                distance = action.throttle * cfg.max_speed * cfg.reverse_factor

            forager.x = wrap_coordinate(
                forager.x + math.cos(forager.heading) * distance,
                self.config.world.width,
            )
            forager.y = wrap_coordinate(
                forager.y + math.sin(forager.heading) * distance,
                self.config.world.height,
            )

            delta_heading = action.turn * cfg.max_turn_rate
            forager.energy -= (
                cfg.basal_cost
                + cfg.movement_cost * abs(distance)
                + cfg.movement_cost * cfg.turn_cost_factor * abs(delta_heading)
            )
            forager.lifetime_steps += 1

    def _consume_food(self) -> None:
        """Resolve food consumption based on distance and energy capacity."""
        width = self.config.world.width
        height = self.config.world.height
        food_energy = self.config.food.energy
        energy_capacity = self.config.forager.energy_capacity

        # Store all valid forager-food contacts as:
        # (distance_squared, forager_index, food_index)
        candidates: list[tuple[float, int, int]] = []

        for forager_index, forager in enumerate(self.foragers):
            # Foragers at energy capacity cannot consume additional food.
            if forager.energy >= energy_capacity:
                continue

            for food_index, resource in enumerate(self.food):
                eat_radius = forager.radius + resource.radius
                distance_squared = toroidal_distance_squared(
                    forager.x,
                    forager.y,
                    resource.x,
                    resource.y,
                    width,
                    height,
                )

                # Only resources inside the physical consumption radius
                # are relevant candidates.
                if distance_squared > eat_radius * eat_radius:
                    continue

                candidates.append((distance_squared, forager_index, food_index))

        # Resolve contacts from nearest to farthest. This avoids making food
        # consumption depend on the current ordering of self.food.
        candidates.sort(key=lambda candidate: candidate[0])

        consumed_food: set[int] = set()

        for _, forager_index, food_index in candidates:
            # A food resource can only be consumed once.
            if food_index in consumed_food:
                continue

            forager = self.foragers[forager_index]

            # An earlier consumption in this step may already have filled
            # the forager's energy capacity.
            if forager.energy >= energy_capacity:
                continue

            forager.energy = min(forager.energy + food_energy, energy_capacity)
            consumed_food.add(food_index)
            self.food_eaten += 1

        # Keep only food resources that were not consumed this step.
        self.food = [
            resource
            for food_index, resource in enumerate(self.food)
            if food_index not in consumed_food
        ]

    def _remove_dead_foragers(self) -> None:
        max_lifetime = self.config.forager.max_lifetime_steps
        living: list[Forager] = []

        for forager in self.foragers:
            lifetime_expired = (
                max_lifetime > 0 and forager.lifetime_steps >= max_lifetime
            )
            if forager.energy <= 0.0 or lifetime_expired:
                self.deaths += 1
            else:
                living.append(forager)

        self.foragers = living

    def _reproduce(self) -> None:
        cfg = self.config.forager
        free_slots = self.config.world.max_population - len(self.foragers)
        if free_slots <= 0:
            return

        eligible = [
            forager
            for forager in self.foragers
            if forager.energy >= cfg.reproduction_threshold
            and forager.lifetime_steps >= cfg.min_reproduction_age_steps
        ]
        if not eligible:
            return

        order = self.rng.permutation(len(eligible))
        children: list[Forager] = []

        for index in order[:free_slots]:
            parent = eligible[int(index)]
            parent.energy -= cfg.reproduction_cost
            children.append(self._make_offspring(parent))

        self.foragers.extend(children)
        self.births += len(children)

    def _spawn_food(self) -> None:
        free_slots = self.config.food.max_count - len(self.food)
        if free_slots <= 0 or self.config.food.spawn_rate <= 0.0:
            return

        spawn_count = min(
            free_slots,
            int(self.rng.poisson(self.config.food.spawn_rate)),
        )
        self.food.extend(self._make_food() for _ in range(spawn_count))
