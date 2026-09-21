# SPDX-License-Identifier: MIT
"""Persistent two-dimensional Predator-Prey simulation."""

import math
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from evolib import EvoNet, Indiv
from evolib.representation.composite import ParaComposite
from evosim.core.geometry import (
    toroidal_displacement,
    toroidal_distance_squared,
    wrap_coordinate,
)
from evosim.core.simulation import Simulation
from evosim.sims.predator_prey.config import AgentModulesConfig, PredatorPreyConfig
from evosim.sims.predator_prey.objects import (
    Agent,
    AgentAction,
    Predator,
    Prey,
    SensorRay,
)


class PredatorPreySimulation(Simulation):
    """Persistent world with independently evolving Prey and Predators."""

    def __init__(self, config: PredatorPreyConfig | str | Path) -> None:
        super().__init__()
        if isinstance(config, (str, Path)):
            config = PredatorPreyConfig.from_yaml(config)
        self.config = config
        self.rng = np.random.default_rng()
        self._prey_rays = self._build_ray_layout(config.modules.prey)
        self._predator_rays = self._build_ray_layout(config.modules.predator)

        self.prey: list[Prey] = []
        self.predators: list[Predator] = []

        self.prey_births = 0
        self.predator_births = 0
        self.predator_deaths = 0
        self.prey_captured = 0

        self.reset(seed=config.seed)

    @property
    def world_size(self) -> tuple[int, int]:
        """Return world dimensions in simulation units."""
        return self.config.world.width, self.config.world.height

    @property
    def prey_population_size(self) -> int:
        """Return the number of living Prey."""
        return len(self.prey)

    @property
    def predator_population_size(self) -> int:
        """Return the number of living Predators."""
        return len(self.predators)

    @property
    def is_extinct(self) -> bool:
        """Return whether either interacting population has gone extinct."""
        return not self.prey or not self.predators

    @property
    def running(self) -> bool:
        """Return whether the configured simulation should continue."""
        return self.step_count < self.config.max_steps and not self.is_extinct

    @property
    def mean_prey_energy(self) -> float:
        """Return mean Prey movement energy, or zero after extinction."""
        if not self.prey:
            return 0.0
        return float(np.mean([prey.energy for prey in self.prey]))

    @property
    def mean_predator_energy(self) -> float:
        """Return mean Predator energy, or zero after extinction."""
        if not self.predators:
            return 0.0
        return float(np.mean([predator.energy for predator in self.predators]))

    def reset(self, *, seed: int | None = None) -> None:
        """Reset populations, counters, and random state."""
        if seed is None:
            seed = self.config.seed

        self.rng = np.random.default_rng(seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.step_count = 0
        self.prey_births = 0
        self.predator_births = 0
        self.predator_deaths = 0
        self.prey_captured = 0

        self.prey = [
            self._make_prey_founder()
            for _ in range(self.config.prey.initial_population)
        ]
        self.predators = [
            self._make_predator_founder()
            for _ in range(self.config.predator.initial_population)
        ]

    def step(self) -> None:
        """Advance one synchronous simulation step."""
        prey_actions = [
            self._calculate_action(prey, self.prey_observation(prey))
            for prey in self.prey
        ]
        predator_actions = [
            self._calculate_action(predator, self.predator_observation(predator))
            for predator in self.predators
        ]

        self._move_prey(prey_actions)
        self._move_predators(predator_actions)
        self._remove_dead_predators()
        self._capture_prey()
        self._reproduce_prey()
        self._reproduce_predators()

        self.step_count += 1

    def metrics(self) -> dict[str, int | float]:
        """Return current simulation metrics without performing I/O."""
        return {
            "step": self.step_count,
            "prey_population": self.prey_population_size,
            "predator_population": self.predator_population_size,
            "prey_births": self.prey_births,
            "predator_births": self.predator_births,
            "predator_deaths": self.predator_deaths,
            "prey_captured": self.prey_captured,
            "mean_prey_energy": self.mean_prey_energy,
            "mean_predator_energy": self.mean_predator_energy,
        }

    @staticmethod
    def _build_ray_layout(modules: AgentModulesConfig) -> tuple[SensorRay, ...]:
        """Build fixed ray centers for one population."""
        spacing = modules.sensor_fov / modules.ray_count
        first_angle = -modules.sensor_fov / 2.0 + spacing / 2.0
        return tuple(
            SensorRay(
                angle=first_angle + index * spacing,
                range=modules.sensor_range,
            )
            for index in range(modules.ray_count)
        )

    def ray_layout(self, agent: Agent) -> tuple[SensorRay, ...]:
        """Return the fixed ray layout for an agent population."""
        if isinstance(agent, Prey):
            return self._prey_rays
        if isinstance(agent, Predator):
            return self._predator_rays
        raise TypeError("Unsupported Predator-Prey agent type.")

    def prey_observation(self, prey: Prey) -> list[float]:
        """Return Predator ray activations plus normalized movement energy."""
        predator_values = self._sense_objects(
            prey,
            self.predators,
            self.ray_layout(prey),
        )
        energy = prey.energy / self.config.prey.energy_capacity
        return [*predator_values, energy]

    def predator_observation(self, predator: Predator) -> list[float]:
        """Return Prey ray activations plus normalized Predator state."""
        prey_values = self._sense_objects(
            predator,
            self.prey,
            self.ray_layout(predator),
        )
        energy = predator.energy / self.config.predator.energy_capacity
        cooldown = (
            predator.feeding_cooldown / self.config.predator.feeding_cooldown_steps
            if self.config.predator.feeding_cooldown_steps > 0
            else 0.0
        )
        return [*prey_values, energy, cooldown]

    def _sense_objects(
        self,
        observer: Agent,
        objects: Sequence[Agent],
        rays: Sequence[SensorRay],
    ) -> list[float]:
        """Return nearest circle-intersection activation for each sensor ray."""
        ray_values = [0.0 for _ in rays]
        ray_directions = [
            (
                math.cos(observer.heading + ray.angle),
                math.sin(observer.heading + ray.angle),
                ray.range,
            )
            for ray in rays
        ]
        max_range = max(ray.range for ray in rays)

        for obj in objects:
            dx, dy = toroidal_displacement(
                observer.x,
                observer.y,
                obj.x,
                obj.y,
                self.config.world.width,
                self.config.world.height,
            )
            distance_squared = dx * dx + dy * dy
            radius_squared = obj.radius * obj.radius

            max_hit_distance = max_range + obj.radius
            if distance_squared > max_hit_distance * max_hit_distance:
                continue

            if distance_squared <= radius_squared:
                ray_values = [1.0 for _ in ray_values]
                continue

            for index, (dir_x, dir_y, ray_range) in enumerate(ray_directions):
                projected = dx * dir_x + dy * dir_y
                if projected <= 0.0:
                    continue

                perpendicular_squared = max(
                    0.0,
                    distance_squared - projected * projected,
                )
                if perpendicular_squared > radius_squared:
                    continue

                half_chord = math.sqrt(radius_squared - perpendicular_squared)
                hit_distance = max(0.0, projected - half_chord)
                if hit_distance > ray_range:
                    continue

                strength = 1.0 - hit_distance / ray_range
                ray_values[index] = max(ray_values[index], strength)

        return ray_values

    @staticmethod
    def _make_indiv(modules: AgentModulesConfig) -> Indiv:
        para = ParaComposite(
            {
                "controller": EvoNet.from_config(modules.controller),
            }
        )
        return Indiv(para)

    def _make_prey_founder(self) -> Prey:
        cfg = self.config.prey
        return Prey(
            indiv=self._make_indiv(self.config.modules.prey),
            x=float(self.rng.uniform(0.0, self.config.world.width)),
            y=float(self.rng.uniform(0.0, self.config.world.height)),
            heading=float(self.rng.uniform(0.0, math.tau)),
            energy=cfg.energy_capacity,
            radius=cfg.radius,
            reproduction_timer=int(
                self.rng.integers(1, cfg.reproduction_interval_steps + 1)
            ),
        )

    def _make_predator_founder(self) -> Predator:
        cfg = self.config.predator
        return Predator(
            indiv=self._make_indiv(self.config.modules.predator),
            x=float(self.rng.uniform(0.0, self.config.world.width)),
            y=float(self.rng.uniform(0.0, self.config.world.height)),
            heading=float(self.rng.uniform(0.0, math.tau)),
            energy=cfg.initial_energy,
            radius=cfg.radius,
        )

    @staticmethod
    def _mutated_indiv(parent: Agent) -> Indiv:
        child_indiv = parent.indiv.copy(
            reset_fitness=True,
            reset_age=True,
            reset_origin=True,
        )
        child_indiv.mutate()
        return child_indiv

    def _prey_offspring_position(self, parent: Prey) -> tuple[float, float]:
        cfg = self.config.prey
        spawn_angle = float(self.rng.uniform(0.0, math.tau))
        spawn_distance = float(
            self.rng.uniform(parent.radius * 3.0, cfg.offspring_dispersion)
        )
        x = wrap_coordinate(
            parent.x + math.cos(spawn_angle) * spawn_distance,
            self.config.world.width,
        )
        y = wrap_coordinate(
            parent.y + math.sin(spawn_angle) * spawn_distance,
            self.config.world.height,
        )
        return x, y

    def _predator_offspring_position(self, parent: Predator) -> tuple[float, float]:
        cfg = self.config.predator
        spawn_angle = float(self.rng.uniform(0.0, math.tau))
        spawn_distance = float(
            self.rng.uniform(parent.radius * 3.0, cfg.offspring_dispersion)
        )
        x = wrap_coordinate(
            parent.x + math.cos(spawn_angle) * spawn_distance,
            self.config.world.width,
        )
        y = wrap_coordinate(
            parent.y + math.sin(spawn_angle) * spawn_distance,
            self.config.world.height,
        )
        return x, y

    def _next_prey_reproduction_delay(self) -> int:
        cfg = self.config.prey
        jitter = cfg.reproduction_interval_jitter_steps
        return int(
            self.rng.integers(
                cfg.reproduction_interval_steps - jitter,
                cfg.reproduction_interval_steps + jitter + 1,
            )
        )

    def _make_prey_offspring(self, parent: Prey) -> Prey:
        cfg = self.config.prey
        x, y = self._prey_offspring_position(parent)
        return Prey(
            indiv=self._mutated_indiv(parent),
            x=x,
            y=y,
            heading=float(self.rng.uniform(0.0, math.tau)),
            energy=cfg.energy_capacity,
            radius=cfg.radius,
            reproduction_timer=self._next_prey_reproduction_delay(),
        )

    def _make_predator_offspring(
        self,
        parent: Predator,
        energy: float,
    ) -> Predator:
        cfg = self.config.predator
        x, y = self._predator_offspring_position(parent)
        return Predator(
            indiv=self._mutated_indiv(parent),
            x=x,
            y=y,
            heading=float(self.rng.uniform(0.0, math.tau)),
            energy=energy,
            radius=cfg.radius,
            reproduction_cooldown=cfg.reproduction_cooldown_steps,
        )

    @staticmethod
    def _composite(agent: Agent) -> ParaComposite:
        para = agent.indiv.para
        if not isinstance(para, ParaComposite):
            raise TypeError(
                "Predator-Prey individuals require a ParaComposite representation."
            )
        return para

    def _controller(self, agent: Agent) -> EvoNet:
        controller = self._composite(agent)["controller"]
        if not isinstance(controller, EvoNet):
            raise TypeError("Predator-Prey requires an EvoNet 'controller' component.")
        return controller

    def _calculate_action(
        self,
        agent: Agent,
        observation: list[float],
    ) -> AgentAction:
        outputs = self._controller(agent).calc(observation)
        if len(outputs) != 2:
            raise ValueError(
                "Predator-Prey controller must return exactly two outputs."
            )

        turn = max(-1.0, min(1.0, float(outputs[0])))
        throttle = max(-1.0, min(1.0, float(outputs[1])))
        return AgentAction(turn=turn, throttle=throttle)

    @staticmethod
    def _movement(
        action: AgentAction,
        max_speed: float,
        max_turn_rate: float,
        reverse_factor: float,
    ) -> tuple[float, float]:
        """Return heading change and signed travel distance for one action."""
        delta_heading = action.turn * max_turn_rate
        if action.throttle >= 0.0:
            distance = action.throttle * max_speed
        else:
            distance = action.throttle * max_speed * reverse_factor
        return delta_heading, distance

    def _move_prey(self, actions: Sequence[AgentAction]) -> None:
        """Move Prey, spend movement energy, and apply stationary grazing."""
        cfg = self.config.prey

        for prey, action in zip(self.prey, actions, strict=True):
            delta_heading, distance = self._movement(
                action,
                cfg.max_speed,
                cfg.max_turn_rate,
                cfg.reverse_factor,
            )

            # Prey energy is a movement budget. The requested translation is
            # limited to the distance that can actually be paid for.
            if cfg.movement_cost > 0.0:
                max_distance = prey.energy / cfg.movement_cost
                distance = max(-max_distance, min(distance, max_distance))

            prey.heading = (prey.heading + delta_heading) % math.tau
            prey.x = wrap_coordinate(
                prey.x + math.cos(prey.heading) * distance,
                self.config.world.width,
            )
            prey.y = wrap_coordinate(
                prey.y + math.sin(prey.heading) * distance,
                self.config.world.height,
            )

            prey.energy = max(
                0.0,
                prey.energy - cfg.movement_cost * abs(distance),
            )

            if abs(distance) <= cfg.grazing_max_speed:
                prey.energy = min(
                    cfg.energy_capacity,
                    prey.energy + cfg.grazing_rate,
                )

    def _move_predators(self, actions: Sequence[AgentAction]) -> None:
        """Move Predators and spend the energy required for survival and motion."""
        cfg = self.config.predator

        for predator, action in zip(self.predators, actions, strict=True):
            delta_heading, distance = self._movement(
                action,
                cfg.max_speed,
                cfg.max_turn_rate,
                cfg.reverse_factor,
            )

            predator.heading = (predator.heading + delta_heading) % math.tau
            predator.x = wrap_coordinate(
                predator.x + math.cos(predator.heading) * distance,
                self.config.world.width,
            )
            predator.y = wrap_coordinate(
                predator.y + math.sin(predator.heading) * distance,
                self.config.world.height,
            )

            predator.energy -= cfg.basal_cost + cfg.movement_cost * abs(distance)

            if predator.feeding_cooldown > 0:
                predator.feeding_cooldown -= 1
            if predator.reproduction_cooldown > 0:
                predator.reproduction_cooldown -= 1

    def _remove_dead_predators(self) -> None:
        """Remove Predators whose energy is depleted."""
        living: list[Predator] = []

        for predator in self.predators:
            if predator.energy <= 0.0:
                self.predator_deaths += 1
            else:
                living.append(predator)

        self.predators = living

    def _capture_prey(self) -> None:
        """Resolve Predator-Prey contacts and feed eligible Predators."""
        cfg = self.config.predator
        candidates: list[tuple[float, int, int]] = []

        for predator_index, predator in enumerate(self.predators):
            if predator.feeding_cooldown > 0:
                continue
            if predator.energy >= cfg.energy_capacity:
                continue

            for prey_index, prey in enumerate(self.prey):
                contact_radius = predator.radius + prey.radius
                distance_squared = toroidal_distance_squared(
                    predator.x,
                    predator.y,
                    prey.x,
                    prey.y,
                    self.config.world.width,
                    self.config.world.height,
                )
                if distance_squared <= contact_radius * contact_radius:
                    candidates.append((distance_squared, predator_index, prey_index))

        candidates.sort(key=lambda candidate: candidate[0])
        captured_prey: set[int] = set()

        for _, predator_index, prey_index in candidates:
            if prey_index in captured_prey:
                continue

            predator = self.predators[predator_index]
            if predator.feeding_cooldown > 0:
                continue
            if predator.energy >= cfg.energy_capacity:
                continue

            predator.energy = min(
                cfg.energy_capacity,
                predator.energy + self.config.prey.energy_value,
            )
            predator.feeding_cooldown = cfg.feeding_cooldown_steps
            captured_prey.add(prey_index)
            self.prey_captured += 1

        if captured_prey:
            self.prey = [
                prey
                for prey_index, prey in enumerate(self.prey)
                if prey_index not in captured_prey
            ]

    def _reproduce_prey(self) -> None:
        """Advance Prey timers and resolve reproduction attempts."""
        cfg = self.config.prey
        due: list[Prey] = []

        for prey in self.prey:
            if prey.reproduction_timer > 0:
                prey.reproduction_timer -= 1

            if prey.reproduction_timer <= 0:
                due.append(prey)
                prey.reproduction_timer = self._next_prey_reproduction_delay()

        free_slots = cfg.max_population - len(self.prey)
        if free_slots <= 0 or not due:
            return

        if len(due) > free_slots:
            order = self.rng.permutation(len(due))[:free_slots]
            parents = [due[int(index)] for index in order]
        else:
            parents = due

        children = [self._make_prey_offspring(parent) for parent in parents]

        self.prey.extend(children)
        self.prey_births += len(children)

    def _reproduce_predators(self) -> None:
        """Split full-energy Predators when their reproduction cooldown elapsed."""
        cfg = self.config.predator
        free_slots = cfg.max_population - len(self.predators)
        if free_slots <= 0:
            return

        eligible = [
            predator
            for predator in self.predators
            if predator.energy >= cfg.energy_capacity
            and predator.reproduction_cooldown == 0
        ]
        if not eligible:
            return

        order = self.rng.permutation(len(eligible))[:free_slots]
        children: list[Predator] = []

        for index in order:
            parent = eligible[int(index)]
            child_energy = parent.energy / 2.0
            parent.energy = child_energy
            parent.reproduction_cooldown = cfg.reproduction_cooldown_steps
            children.append(self._make_predator_offspring(parent, child_energy))

        self.predators.extend(children)
        self.predator_births += len(children)
