# SPDX-License-Identifier: MIT
"""Headless Jumper environment with one ray-based obstacle sensor."""

import math
import random
from collections.abc import Iterator

import pygame
from evoenv.core.env import Action, Env, InfoDict, Observation, StepResult
from evoenv.core.sensors import Pose2D, RaySensor, SensorLineState
from evoenv.envs.jumper_defaults import (
    DEFAULT_GROUND_Y_OFFSET,
    DEFAULT_PLAYER_X_OFFSET,
)
from evoenv.envs.jumper_objects import JumperObstacle, JumperPlayer


class JumperEnv(Env):
    """
    Side-scrolling jump task with one obstacle sensor.

    Observation:
    - sensor_value: Generic ray sensor value in [0.0, 1.0]. A value of 0.0 means
      inactive or no obstacle hit. A value close to 1.0 means a strong nearby hit.
    - normalized_obstacle_height: Height of the nearest obstacle in front of the player,
      normalized to [0.0, 1.0] using the configured height range.

    Action:
    - jump_signal: Jump trigger. Values greater than 0.5 request a jump.
    - jump_strength: Jump impulse strength in [0.0, 1.0].
    """

    observation_size = 2
    action_size = 2

    def __init__(
        self,
        *,
        width: int,
        height: int,
        max_steps: int,
        gravity: float,
        jump_velocity: float,
        obstacle_speed: float,
        obstacle_width: int,
        min_obstacle_height: int,
        max_obstacle_height: int,
        min_spawn_gap: int,
        max_spawn_gap: int,
        terminate_on_collision: bool,
        collision_penalty: float,
        pass_reward: float,
        alive_reward: float,
        jump_strength_penalty: float,
        sensor: RaySensor,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.max_steps = int(max_steps)
        self.ground_y = self.height - DEFAULT_GROUND_Y_OFFSET
        self.player_x = float(DEFAULT_PLAYER_X_OFFSET)

        self.gravity = float(gravity)
        self.jump_velocity = float(jump_velocity)

        self.obstacle_speed = float(obstacle_speed)
        self.obstacle_width = int(obstacle_width)
        self.min_obstacle_height = int(min_obstacle_height)
        self.max_obstacle_height = int(max_obstacle_height)
        self.min_spawn_gap = int(min_spawn_gap)
        self.max_spawn_gap = int(max_spawn_gap)

        self.terminate_on_collision = bool(terminate_on_collision)
        self.collision_penalty = float(collision_penalty)
        self.pass_reward = float(pass_reward)
        self.alive_reward = float(alive_reward)
        self.jump_strength_penalty = float(jump_strength_penalty)

        self.sensor = sensor

        self.player = JumperPlayer(
            x=self.player_x,
            ground_y=self.ground_y,
            gravity=self.gravity,
            jump_velocity=self.jump_velocity,
        )

        self.obstacle_group = pygame.sprite.Group()

        self.step_count = 0
        self.passed_obstacles = 0
        self.collision_count = 0
        self._rng = random.Random()

    def reset(self, seed: int | None = None) -> Observation:
        """Reset the episode and return the initial observation."""
        if seed is not None:
            self._rng.seed(seed)

        self.step_count = 0
        self.passed_obstacles = 0
        self.collision_count = 0
        self.obstacle_group.empty()

        self.player.reset(x=self.player_x, ground_y=self.ground_y)
        self._spawn_initial_obstacle()

        return self._observe()

    def step(self, action: Action) -> StepResult:
        """Advance the simulation by one step."""
        jump_signal = max(0.0, min(1.0, float(action[0])))
        jump_strength = max(0.0, min(1.0, float(action[1])))

        did_jump = self.player.step(
            jump_signal=jump_signal,
            jump_strength=jump_strength,
        )
        self.obstacle_group.update()
        self._spawn_if_needed()

        self.step_count += 1

        passed_obstacle = self._mark_passed_obstacles()
        has_collision = self._has_collision()

        if has_collision:
            self.collision_count += 1

        reward = self.alive_reward
        if passed_obstacle:
            reward += self.pass_reward
        if has_collision:
            reward -= self.collision_penalty
        if did_jump:
            reward -= (jump_strength**2) * self.jump_strength_penalty

        done = self.step_count >= self.max_steps
        if has_collision and self.terminate_on_collision:
            done = True

        info: InfoDict = {
            "x": self.player.x,
            "y": self.player.y,
            "jump_signal": jump_signal,
            "jump_strength": jump_strength,
            "player_height": self.player.normalized_height,
            "on_ground": self.player.on_ground,
            "sensor_value": self._sensor_value(),
            "normalized_obstacle_height": self._normalized_nearest_obstacle_height(),
            "nearest_obstacle_distance": self._nearest_obstacle_distance(),
            "passed_obstacles": self.passed_obstacles,
            "passed_obstacle": passed_obstacle,
            "collision": self.collision_count,
            "has_collision": has_collision,
        }

        return self._observe(), reward, done, info

    def get_sensor_states(self) -> list[SensorLineState]:
        """Return current sensor state for rendering and debugging."""
        value, hit_fraction = self._cast_sensor()
        return [
            self.sensor.get_state(
                self._sensor_pose(),
                value=value,
                hit_fraction=hit_fraction,
            )
        ]

    def _observe(self) -> Observation:
        """Return the fixed two-value observation for the EvoNet controller."""
        return [
            self._sensor_value(),
            self._normalized_nearest_obstacle_height(),
        ]

    def _sensor_value(self) -> float:
        """Return the generic proximity value of the configured obstacle sensor."""
        value, _hit_fraction = self._cast_sensor()
        return value

    def _cast_sensor(self) -> tuple[float, float | None]:
        """Cast the ray sensor and return sensor value plus hit fraction."""
        state = self.sensor.get_state(self._sensor_pose(), value=0.0)
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

    def _spawn_initial_obstacle(self) -> None:
        """Spawn the first obstacle with a slightly shorter initial gap."""
        self._spawn_obstacle(
            x=self.width
            + self._rng.uniform(
                self.min_spawn_gap * 0.25,
                self.max_spawn_gap * 0.40,
            )
        )

    def _spawn_if_needed(self) -> None:
        """Spawn a new obstacle if the rightmost obstacle moved onto the screen."""
        obstacles = list(self._iter_obstacles())
        if not obstacles:
            self._spawn_obstacle(
                x=self.width
                + self._rng.uniform(
                    self.min_spawn_gap,
                    self.max_spawn_gap,
                )
            )
            return

        rightmost_x = max(float(obstacle.rect.centerx) for obstacle in obstacles)
        if rightmost_x < self.width:
            self._spawn_obstacle(
                x=self.width
                + self._rng.uniform(
                    self.min_spawn_gap,
                    self.max_spawn_gap,
                )
            )

    def _spawn_obstacle(self, *, x: float) -> None:
        """Spawn one rectangular obstacle with variable height."""
        obstacle_height = self._rng.randint(
            self.min_obstacle_height,
            self.max_obstacle_height,
        )

        self.obstacle_group.add(
            JumperObstacle(
                x=float(x),
                ground_y=float(self.ground_y),
                speed=self.obstacle_speed,
                width=self.obstacle_width,
                height=obstacle_height,
            )
        )

    def _mark_passed_obstacles(self) -> bool:
        """Mark passed obstacles and return True if at least one was newly passed."""
        passed_obstacle = False
        for obstacle in self._iter_obstacles():
            if obstacle.counted:
                continue

            if obstacle.rect.right < self.player.rect.left:
                obstacle.counted = True
                self.passed_obstacles += 1
                passed_obstacle = True

        return passed_obstacle

    def _has_collision(self) -> bool:
        """Return True if the player overlaps any obstacle."""
        return bool(
            pygame.sprite.spritecollide(
                self.player,
                self.obstacle_group,
                dokill=False,
            )
        )

    def _nearest_obstacle(self) -> JumperObstacle | None:
        """Return the nearest obstacle that is not fully behind the player."""
        candidates = [
            obstacle
            for obstacle in self._iter_obstacles()
            if obstacle.rect.right >= self.player.rect.left
        ]

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda obstacle: max(
                0.0, float(obstacle.rect.left - self.player.rect.right)
            ),
        )

    def _normalized_nearest_obstacle_height(self) -> float:
        """Return normalized height of the nearest obstacle in front of the player."""
        obstacle = self._nearest_obstacle()
        if obstacle is None:
            return 0.0

        min_height = float(self.min_obstacle_height)
        max_height = float(self.max_obstacle_height)
        height_range = max(1.0, max_height - min_height)

        return max(0.0, min(1.0, (float(obstacle.height) - min_height) / height_range))

    def _nearest_obstacle_distance(self) -> float:
        """Return normalized distance to the closest obstacle in front of the player."""
        obstacle = self._nearest_obstacle()
        if obstacle is None:
            return 1.0

        distance = max(0.0, float(obstacle.rect.left - self.player.rect.right))
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
