# SPDX-License-Identifier: MIT
"""Headless Jumper environment with ray-based obstacle sensing."""

import math
import random
from collections.abc import Iterator

import pygame
from evoenv.core.env import Action, Env, InfoDict, Observation, StepResult
from evoenv.core.sensors import Pose2D, RaySensor, SensorLineState
from evoenv.envs.jumper_defaults import (
    DEFAULT_GROUND_Y_OFFSET,
    DEFAULT_HEIGHT,
    DEFAULT_MAX_STEPS,
    DEFAULT_PLAYER_X_OFFSET,
    DEFAULT_WIDTH,
)
from evoenv.envs.jumper_objects import JumperObstacle, JumperPlayer
from evoenv.envs.jumper_settings import get_jumper_settings

SensorLayout = tuple[RaySensor, ...]


class JumperEnv(Env):
    """Side-scrolling jump task with externally supplied ray sensors."""

    action_size = 2

    def __init__(
        self,
        *,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        max_steps: int = DEFAULT_MAX_STEPS,
        difficulty: str = "medium",
        sensors: SensorLayout | None = None,
    ) -> None:
        self.settings = get_jumper_settings(difficulty)

        self.width = int(width)
        self.height = int(height)
        self.max_steps = int(max_steps)
        self.ground_y = self.height - DEFAULT_GROUND_Y_OFFSET
        self.player_x = float(DEFAULT_PLAYER_X_OFFSET)

        self.sensors: SensorLayout = sensors or self.default_sensors()
        if not self.sensors:
            raise ValueError("JumperEnv requires at least one sensor.")

        self.observation_size = len(self.sensors) + 2
        self.action_size = 2

        self.player = JumperPlayer(
            x=self.player_x,
            ground_y=self.ground_y,
            gravity=self.settings.gravity,
            jump_velocity=self.settings.jump_velocity,
        )

        self.obstacle_group = pygame.sprite.Group()

        self.step_count = 0
        self.passed_obstacles = 0
        self.collision_count = 0
        self._next_spawn_x = 0.0
        self._rng = random.Random()

    @staticmethod
    def default_sensors() -> SensorLayout:
        """Return the deterministic default Jumper sensor layout."""
        return (RaySensor(length=250.0, angle=math.pi / 2.0),)

    def reset(self, seed: int | None = None) -> Observation:
        """Reset the episode and return the initial observation."""
        if seed is not None:
            self._rng.seed(seed)

        self.step_count = 0
        self.passed_obstacles = 0
        self.collision_count = 0
        self.obstacle_group.empty()

        self.player.reset(x=self.player_x, ground_y=self.ground_y)
        self._next_spawn_x = self.width + self._rng.uniform(
            self.settings.min_spawn_gap * 0.25,
            self.settings.max_spawn_gap * 0.40,
        )
        self._spawn_obstacle(x=self._next_spawn_x)

        return self._observe()

    def step(self, action: Action) -> StepResult:
        """Advance the simulation by one step."""
        if len(action) != self.action_size:
            raise ValueError(
                f"Expected action with {self.action_size} values, got {len(action)}."
            )

        jump_signal = max(0.0, min(1.0, float(action[0])))
        jump_force = max(0.0, min(1.0, float(action[1])))

        self.player.step(jump_signal=jump_signal, jump_force=jump_force)
        self.obstacle_group.update()
        self._spawn_if_needed()

        self.step_count += 1

        passed_obstacle = False
        for obstacle in self._iter_obstacles():
            if not obstacle.counted and obstacle.rect.right < self.player.rect.left:
                obstacle.counted = True
                self.passed_obstacles += 1
                passed_obstacle = True

        has_collision = self._has_collision()
        if has_collision:
            self.collision_count += 1

        reward = self.settings.alive_reward
        if passed_obstacle:
            reward += self.settings.pass_reward
        if has_collision:
            reward -= self.settings.collision_penalty

        done = self.step_count >= self.max_steps
        if has_collision and self.settings.terminate_on_collision:
            done = True

        observation = self._observe()
        obstacle_distance = self._nearest_obstacle_distance()

        info: InfoDict = {
            "x": self.player.x,
            "y": self.player.y,
            "jump_signal": jump_signal,
            "jump_force": jump_force,
            "player_height": self.player.normalized_height,
            "on_ground": self.player.on_ground,
            "nearest_obstacle_distance": obstacle_distance,
            "passed_obstacles": self.passed_obstacles,
            "passed_obstacle": passed_obstacle,
            "collision": self.collision_count,
            "has_collision": has_collision,
            "difficulty": self.settings.difficulty.value,
        }

        return observation, reward, done, info

    def get_sensor_states(self) -> list[SensorLineState]:
        """Return current sensor states for rendering and debugging."""
        states: list[SensorLineState] = []

        for sensor in self.sensors:
            value, hit_fraction = self._cast_sensor(sensor)
            states.append(
                sensor.get_state(
                    self._sensor_pose(),
                    value=value,
                    hit_fraction=hit_fraction,
                )
            )

        return states

    def _observe(self) -> Observation:
        """Return fixed-size observation values for the EvoNet controller."""
        sensor_values = [self._sensor_value(sensor) for sensor in self.sensors]

        return [
            *sensor_values,
            self.player.normalized_height,
            1.0 if self.player.on_ground else 0.0,
        ]

    def _sensor_value(self, sensor: RaySensor) -> float:
        """Return proximity value for the first obstacle hit by one sensor ray."""
        value, _hit_fraction = self._cast_sensor(sensor)
        return value

    def _cast_sensor(self, sensor: RaySensor) -> tuple[float, float | None]:
        """Cast one sensor ray and return proximity value plus hit fraction."""
        state = sensor.get_state(self._sensor_pose(), value=0.0)
        first_hit_fraction: float | None = None

        for obstacle in self._iter_obstacles():
            hit_fraction = self._ray_rect_hit_fraction(
                start_x=state.start_x,
                start_y=state.start_y,
                end_x=state.end_x,
                end_y=state.end_y,
                rect=obstacle.rect,
            )
            if hit_fraction is None:
                continue

            if first_hit_fraction is None or hit_fraction < first_hit_fraction:
                first_hit_fraction = hit_fraction

        if first_hit_fraction is None:
            return 0.0, None

        return 1.0 - first_hit_fraction, first_hit_fraction

    def _sensor_pose(self) -> Pose2D:
        """Return the player-front pose used as the Jumper sensor origin."""
        return Pose2D(
            x=float(self.player.rect.right),
            y=float(self.player.rect.centery),
            heading=0.0,
        )

    def _spawn_if_needed(self) -> None:
        """Spawn a new obstacle if the rightmost obstacle moved far enough left."""
        obstacles = list(self._iter_obstacles())
        if not obstacles:
            self._spawn_obstacle(
                x=self.width
                + self._rng.uniform(
                    self.settings.min_spawn_gap,
                    self.settings.max_spawn_gap,
                )
            )
            return

        rightmost_x = max(float(obstacle.rect.centerx) for obstacle in obstacles)
        if rightmost_x < self.width:
            self._spawn_obstacle(
                x=self.width
                + self._rng.uniform(
                    self.settings.min_spawn_gap,
                    self.settings.max_spawn_gap,
                )
            )

    def _spawn_obstacle(self, *, x: float) -> None:
        """Spawn one rectangular obstacle."""
        self.obstacle_group.add(
            JumperObstacle(
                x=float(x),
                ground_y=float(self.ground_y),
                speed=self.settings.obstacle_speed,
                width=self.settings.obstacle_width,
                height=self.settings.obstacle_height,
            )
        )

    def _has_collision(self) -> bool:
        """Return True if the player overlaps any obstacle."""
        return bool(
            pygame.sprite.spritecollide(
                self.player,
                self.obstacle_group,
                dokill=False,
            )
        )

    def _nearest_obstacle_distance(self) -> float:
        """Return normalized distance to the closest obstacle in front of the player."""
        candidates = [
            float(obstacle.rect.left - self.player.rect.right)
            for obstacle in self._iter_obstacles()
            if obstacle.rect.right >= self.player.rect.left
        ]

        if not candidates:
            return 1.0

        distance = max(0.0, min(candidates))
        return min(1.0, distance / max(1.0, float(self.width)))

    def _iter_obstacles(self) -> Iterator[JumperObstacle]:
        """Yield active Jumper obstacles."""
        for sprite in self.obstacle_group:
            if isinstance(sprite, JumperObstacle):
                yield sprite

    @staticmethod
    def _ray_rect_hit_fraction(
        *,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        rect: pygame.Rect,
    ) -> float | None:
        """Return first ray fraction intersecting one rectangle."""
        ray_length = math.hypot(end_x - start_x, end_y - start_y)
        if ray_length <= 0.0:
            return None

        start = (int(round(start_x)), int(round(start_y)))
        end = (int(round(end_x)), int(round(end_y)))

        clipped_line = rect.clipline(start, end)
        if not clipped_line:
            return None

        hit_x, hit_y = clipped_line[0]
        distance = math.hypot(float(hit_x) - start_x, float(hit_y) - start_y)

        return max(0.0, min(1.0, distance / ray_length))
