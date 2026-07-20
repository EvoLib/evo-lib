# SPDX-License-Identifier: MIT
"""Headless Collector environment for exploration and reward shaping."""

from __future__ import annotations

import math
import random
from collections.abc import Iterator

import pygame
from evoenv.core.env import Action, Env, InfoDict, Observation, StepResult
from evoenv.core.sensors import (
    Pose2D,
    RaySensor,
    SensorLineState,
    ray_rect_hit_fraction,
)
from evoenv.core.utils import clamp, clamp01
from evoenv.envs.collector_objects import (
    CollectorAgent,
    CollectorFood,
    CollectorObstacle,
    circle_intersects_rect,
    distance,
)


class CollectorEnv(Env):
    """
    2D food collection task with obstacle sensors and shaped rewards.

    Observation:
    - target_angle_sin: Sine of the relative angle to the nearest food item.
    - target_angle_cos: Cosine of the relative angle to the nearest food item.
    - target_distance: Distance to nearest food normalized to the arena diagonal.
    - obstacle ray values: One value per configured ray sensor.

    Action:
    - turn: Heading change request in [-1.0, 1.0].
    - throttle: Forward speed request in [0.0, 1.0].
    """

    action_size = 2

    def __init__(
        self,
        *,
        width: int,
        height: int,
        max_steps: int,
        agent_radius: int,
        base_speed: float,
        turn_strength: float,
        food_count: int,
        food_radius: int,
        collect_radius: float,
        obstacle_count: int,
        obstacle_min_size: int,
        obstacle_max_size: int,
        spawn_margin: int,
        terminate_on_collision: bool,
        food_reward: float,
        distance_reward: float,
        exploration_reward: float,
        collision_penalty: float,
        step_penalty: float,
        turn_penalty: float,
        ray_length: float,
        ray_angles: list[float],
        exploration_cell_size: int,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.max_steps = int(max_steps)
        self.agent_radius = int(agent_radius)
        self.base_speed = float(base_speed)
        self.turn_strength = float(turn_strength)

        self.food_count = int(food_count)
        self.food_radius = int(food_radius)
        self.collect_radius = float(collect_radius)

        self.obstacle_count = int(obstacle_count)
        self.obstacle_min_size = int(obstacle_min_size)
        self.obstacle_max_size = int(obstacle_max_size)
        self.spawn_margin = int(spawn_margin)
        self.terminate_on_collision = bool(terminate_on_collision)

        self.food_reward = float(food_reward)
        self.distance_reward = float(distance_reward)
        self.exploration_reward = float(exploration_reward)
        self.collision_penalty = float(collision_penalty)
        self.step_penalty = float(step_penalty)
        self.turn_penalty = float(turn_penalty)

        self.sensors = [
            RaySensor(length=ray_length, angle=angle) for angle in ray_angles
        ]
        self.exploration_cell_size = int(exploration_cell_size)
        self.observation_size = 3 + len(self.sensors)

        self.step_count = 0
        self.food_collected = 0
        self.collision_count = 0
        self.visited_cells: set[tuple[int, int]] = set()
        self.agent = CollectorAgent(
            x=self.width / 2.0,
            y=self.height / 2.0,
            heading=0.0,
            radius=self.agent_radius,
        )
        self.food_items: list[CollectorFood] = []
        self.obstacles: list[CollectorObstacle] = []
        self._rng = random.Random()
        self._arena_diagonal = math.hypot(float(self.width), float(self.height))

    def reset(self, seed: int | None = None) -> Observation:
        """Reset the episode and return the initial observation."""
        if seed is not None:
            self._rng.seed(seed)

        self.step_count = 0
        self.food_collected = 0
        self.collision_count = 0
        self.visited_cells.clear()

        self.agent = CollectorAgent(
            x=self.width / 2.0,
            y=self.height / 2.0,
            heading=self._rng.uniform(0.0, math.tau),
            radius=self.agent_radius,
        )

        self.obstacles = self._spawn_obstacles()
        self.food_items = self._spawn_food_items()
        self._mark_current_cell_visited()

        return self._observe()

    def step(self, action: Action) -> StepResult:
        """Advance the simulation by one step."""
        turn = clamp(action[0], -1.0, 1.0)
        throttle = clamp01(action[1])

        previous_distance = self._nearest_food_distance()
        previous_position = self.agent.position
        previous_heading = self.agent.heading

        self.agent.move(
            turn=turn,
            throttle=throttle,
            base_speed=self.base_speed,
            turn_strength=self.turn_strength,
        )
        self.step_count += 1

        has_collision = self._has_collision()
        if has_collision:
            self.collision_count += 1
            self.agent.x, self.agent.y = previous_position
            self.agent.heading = previous_heading

        current_distance = self._nearest_food_distance()
        collected_now = self._collect_food()

        reward = -self.step_penalty
        reward -= abs(turn) * self.turn_penalty

        if has_collision:
            reward -= self.collision_penalty

        if collected_now > 0:
            reward += collected_now * self.food_reward

        if previous_distance is not None and current_distance is not None:
            progress = (previous_distance - current_distance) / self.base_speed
            reward += progress * self.distance_reward

        if self._mark_current_cell_visited():
            reward += self.exploration_reward

        done = self.step_count >= self.max_steps or not self.food_items
        if has_collision and self.terminate_on_collision:
            done = True

        info: InfoDict = {
            "x": self.agent.x,
            "y": self.agent.y,
            "heading": self.agent.heading,
            "turn": turn,
            "throttle": throttle,
            "food_left": len(self.food_items),
            "food_collected": self.food_collected,
            "collected_now": collected_now,
            "collision": self.collision_count,
            "has_collision": has_collision,
            "visited_cells": len(self.visited_cells),
        }

        return self._observe(), reward, done, info

    def get_sensor_states(self) -> list[SensorLineState]:
        """Return current sensor states for rendering and debugging."""
        pose = self._sensor_pose()
        states: list[SensorLineState] = []
        for sensor in self.sensors:
            value, hit_fraction = self._cast_sensor(sensor)
            states.append(
                sensor.get_state(
                    pose,
                    value=value,
                    hit_fraction=hit_fraction,
                )
            )
        return states

    def _observe(self) -> Observation:
        """Return the current flat observation vector."""
        relative_angle, target_distance = self._target_state()
        obstacle_values = [self._sensor_value(sensor) for sensor in self.sensors]
        return [
            math.sin(relative_angle),
            math.cos(relative_angle),
            target_distance,
            *obstacle_values,
        ]

    def _target_state(self) -> tuple[float, float]:
        """Return relative angle and normalized distance to the nearest food item."""
        nearest_food = self._nearest_food()
        if nearest_food is None:
            return 0.0, 0.0

        dx = nearest_food.x - self.agent.x
        dy = nearest_food.y - self.agent.y
        absolute_angle = math.atan2(dx, -dy)
        relative_angle = self._normalize_angle(absolute_angle - self.agent.heading)
        target_distance = min(
            1.0,
            math.hypot(dx, dy) / max(1.0, self._arena_diagonal),
        )
        return relative_angle, target_distance

    def _nearest_food(self) -> CollectorFood | None:
        """Return the nearest remaining food item."""
        if not self.food_items:
            return None

        return min(
            self.food_items,
            key=lambda food: distance(self.agent.position, food.position),
        )

    def _nearest_food_distance(self) -> float | None:
        """Return distance to the nearest remaining food item."""
        nearest_food = self._nearest_food()
        if nearest_food is None:
            return None

        return distance(self.agent.position, nearest_food.position)

    def _collect_food(self) -> int:
        """Remove collected food items and return the number collected this step."""
        remaining_food: list[CollectorFood] = []
        collected_now = 0

        for food in self.food_items:
            if distance(self.agent.position, food.position) <= self.collect_radius:
                collected_now += 1
                continue

            remaining_food.append(food)

        self.food_items = remaining_food
        self.food_collected += collected_now
        return collected_now

    def _has_collision(self) -> bool:
        """Return True if the agent touches a wall or an obstacle."""
        if self.agent.x - self.agent.radius < 0.0:
            return True
        if self.agent.x + self.agent.radius > float(self.width):
            return True
        if self.agent.y - self.agent.radius < 0.0:
            return True
        if self.agent.y + self.agent.radius > float(self.height):
            return True

        return any(
            circle_intersects_rect(
                x=self.agent.x,
                y=self.agent.y,
                radius=float(self.agent.radius),
                rect=obstacle.rect,
            )
            for obstacle in self.obstacles
        )

    def _mark_current_cell_visited(self) -> bool:
        """Mark the current exploration cell and return True if it was new."""
        cell = (
            int(self.agent.x // self.exploration_cell_size),
            int(self.agent.y // self.exploration_cell_size),
        )
        if cell in self.visited_cells:
            return False

        self.visited_cells.add(cell)
        return True

    def _sensor_value(self, sensor: RaySensor) -> float:
        """Return the proximity value for one obstacle sensor."""
        value, _hit_fraction = self._cast_sensor(sensor)
        return value

    def _cast_sensor(self, sensor: RaySensor) -> tuple[float, float | None]:
        """Cast one ray sensor and return sensor value plus hit fraction."""
        state = sensor.get_state(self._sensor_pose(), value=0.0)
        first_hit_fraction: float | None = None

        for rect in self._iter_blocking_rects():
            hit_fraction = ray_rect_hit_fraction(
                start_x=state.start_x,
                start_y=state.start_y,
                end_x=state.end_x,
                end_y=state.end_y,
                rect=rect,
            )
            if hit_fraction is None:
                continue

            if first_hit_fraction is None or hit_fraction < first_hit_fraction:
                first_hit_fraction = hit_fraction

        if first_hit_fraction is None:
            return 0.0, None

        return 1.0 - first_hit_fraction, first_hit_fraction

    def _sensor_pose(self) -> Pose2D:
        """Return the sensor origin pose at the agent center."""
        return Pose2D(x=self.agent.x, y=self.agent.y, heading=self.agent.heading)

    def _iter_blocking_rects(self) -> Iterator[pygame.Rect]:
        """Yield obstacles and boundary walls as ray-blocking rectangles."""
        for obstacle in self.obstacles:
            yield obstacle.rect

        wall_thickness = 10
        yield pygame.Rect(-wall_thickness, 0, wall_thickness, self.height)
        yield pygame.Rect(self.width, 0, wall_thickness, self.height)
        yield pygame.Rect(0, -wall_thickness, self.width, wall_thickness)
        yield pygame.Rect(0, self.height, self.width, wall_thickness)

    def _spawn_obstacles(self) -> list[CollectorObstacle]:
        """Create random rectangular obstacles."""
        obstacles: list[CollectorObstacle] = []
        protected_center = pygame.Rect(0, 0, 140, 140)
        protected_center.center = (int(round(self.agent.x)), int(round(self.agent.y)))

        for _ in range(self.obstacle_count):
            for _attempt in range(100):
                width = self._rng.randint(
                    self.obstacle_min_size,
                    self.obstacle_max_size,
                )
                height = self._rng.randint(
                    self.obstacle_min_size,
                    self.obstacle_max_size,
                )
                x = self._rng.randint(
                    self.spawn_margin,
                    max(self.spawn_margin, self.width - self.spawn_margin - width),
                )
                y = self._rng.randint(
                    self.spawn_margin,
                    max(self.spawn_margin, self.height - self.spawn_margin - height),
                )
                rect = pygame.Rect(x, y, width, height)

                if rect.colliderect(protected_center):
                    continue

                if any(
                    rect.colliderect(obstacle.rect.inflate(20, 20))
                    for obstacle in obstacles
                ):
                    continue

                obstacles.append(CollectorObstacle(rect=rect))
                break

        return obstacles

    def _spawn_food_items(self) -> list[CollectorFood]:
        """Create random food items outside obstacles."""
        food_items: list[CollectorFood] = []
        min_x = self.spawn_margin + self.food_radius
        max_x = self.width - self.spawn_margin - self.food_radius
        min_y = self.spawn_margin + self.food_radius
        max_y = self.height - self.spawn_margin - self.food_radius

        for _ in range(self.food_count):
            for _attempt in range(200):
                x = self._rng.uniform(float(min_x), float(max_x))
                y = self._rng.uniform(float(min_y), float(max_y))

                if distance((x, y), self.agent.position) < 60.0:
                    continue

                if any(
                    circle_intersects_rect(
                        x=x,
                        y=y,
                        radius=float(self.food_radius + 4),
                        rect=obstacle.rect,
                    )
                    for obstacle in self.obstacles
                ):
                    continue

                food_items.append(CollectorFood(x=x, y=y, radius=self.food_radius))
                break

        return food_items

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize an angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= math.tau
        while angle < -math.pi:
            angle += math.tau
        return angle
