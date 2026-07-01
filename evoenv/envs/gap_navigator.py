# SPDX-License-Identifier: MIT
"""Headless GapNavigator environment with falling obstacle rows."""

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
from evoenv.core.utils import clamp
from evoenv.envs.gap_navigator_objects import (
    AvoiderPlayer,
    GapRow,
    ObstacleBlockSprite,
)

SensorLayout = tuple[RaySensor, ...]


class GapNavigatorEnv(Env):
    """Obstacle row avoidance environment with externally supplied sensors."""

    action_size = 1

    def __init__(
        self,
        *,
        width: int,
        height: int,
        max_steps: int,
        max_sensors: int,
        player_y_offset: int,
        player_speed: float,
        row_speed: float,
        row_interval: int,
        obstacle_height: int,
        min_gap_width: float,
        max_gap_width: float,
        edge_margin: float,
        terminate_on_collision: bool,
        sensors: SensorLayout | None = None,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.max_steps = int(max_steps)
        self.max_sensors = int(max_sensors)

        self.player_y_offset = int(player_y_offset)
        self.player_speed = float(player_speed)

        self.row_speed = float(row_speed)
        self.row_interval = int(row_interval)
        self.obstacle_height = int(obstacle_height)
        self.min_gap_width = float(min_gap_width)
        self.max_gap_width = float(max_gap_width)
        self.edge_margin = float(edge_margin)
        self.terminate_on_collision = bool(terminate_on_collision)

        self.observation_size = self.max_sensors + 2
        self.action_size = 1

        self.sensors: SensorLayout = sensors or self.default_sensors(
            max_sensors=self.max_sensors,
        )

        if len(self.sensors) != self.max_sensors:
            raise ValueError(
                "Sensor layout size mismatch: "
                f"expected {self.max_sensors}, got {len(self.sensors)}."
            )

        self.player = AvoiderPlayer(
            x=self.width * 0.5,
            y=self.height - self.player_y_offset,
            speed=self.player_speed,
        )

        self.gap_rows: list[GapRow] = []
        self.block_group = pygame.sprite.Group()

        self.step_count = 0
        self.passed_rows = 0
        self.collision = 0
        self._rng = random.Random()

    @staticmethod
    def default_sensors(*, max_sensors: int) -> SensorLayout:
        """Return a deterministic hand-designed fallback sensor layout."""
        if max_sensors <= 1:
            return (RaySensor(length=220.0, angle=0.0),)

        center = (max_sensors - 1) / 2.0
        return tuple(
            RaySensor(
                length=220.0,
                angle=(index - center) * 0.28,
            )
            for index in range(max_sensors)
        )

    def reset(self, seed: int | None = None) -> Observation:
        """Reset the episode and return the initial observation."""
        if seed is not None:
            self._rng.seed(seed)

        self.step_count = 0
        self.passed_rows = 0
        self.collision = 0
        self.gap_rows.clear()
        self.block_group.empty()

        self.player.reset(
            x=self.width * 0.5,
            y=self.height - self.player_y_offset,
        )

        self._spawn_row(y=-float(self.obstacle_height))

        return self._observe()

    def step(self, action: Action) -> StepResult:
        """Advance the simulation by one step."""
        clamped_steering = clamp(action[0], -1.0, 1.0)

        self.player.step(
            clamped_steering,
            min_x=self.player.width / 2,
            max_x=self.width - self.player.width / 2,
        )

        for row in self.gap_rows:
            row.step()

        self.block_group.update()
        self._remove_inactive_rows()

        self.step_count += 1

        if self.step_count % self.row_interval == 0:
            self._spawn_row(y=-float(self.obstacle_height))

        row_passed = False
        for row in self.gap_rows:
            if self._row_passed(row) and not row.counted:
                row.counted = True
                self.passed_rows += 1
                row_passed = True

        has_collision = self._has_collision()
        if has_collision:
            self.collision += 1

        done = self.step_count >= self.max_steps
        if has_collision and self.terminate_on_collision:
            done = True

        info: InfoDict = {
            "x": self.player.x,
            "steering": clamped_steering,
            "passed_rows": self.passed_rows,
            "collision": self.collision,
            "has_collision": has_collision,
            "row_passed": row_passed,
            "gap_alignment": self._gap_alignment_score(),
            "near_wall": self._is_near_wall(),
        }

        return self._observe(), 0.0, done, info

    def get_sensor_states(self) -> list[SensorLineState]:
        """Return current sensor states for rendering and debugging."""
        pose = Pose2D(
            x=self.player.x,
            y=self.player.y,
            heading=0.0,
        )

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
        """Return fixed-size observation values for the EvoNet controller."""
        sensor_values = [self._sensor_value(sensor) for sensor in self.sensors]

        if len(sensor_values) != self.max_sensors:
            raise ValueError(
                "Sensor layout size mismatch: "
                f"expected {self.max_sensors}, got {len(sensor_values)}."
            )

        normalized_x = self.player.x / self.width
        normalized_velocity_x = self.player.velocity_x / self.player.speed

        return [
            *sensor_values,
            normalized_x,
            normalized_velocity_x,
        ]

    def _sensor_value(self, sensor: RaySensor) -> float:
        """Return proximity value for the first obstacle hit by one sensor ray."""
        value, _hit_fraction = self._cast_sensor(sensor)
        return value

    def _cast_sensor(self, sensor: RaySensor) -> tuple[float, float | None]:
        """Cast one sensor ray and return proximity value plus hit fraction."""
        pose = Pose2D(
            x=self.player.x,
            y=self.player.y,
            heading=0.0,
        )
        state = sensor.get_state(pose, value=0.0)

        first_hit_fraction: float | None = None

        for block in self._iter_blocks():
            hit_fraction = ray_rect_hit_fraction(
                start_x=state.start_x,
                start_y=state.start_y,
                end_x=state.end_x,
                end_y=state.end_y,
                rect=block.rect,
            )
            if hit_fraction is None:
                continue

            if first_hit_fraction is None or hit_fraction < first_hit_fraction:
                first_hit_fraction = hit_fraction

        if first_hit_fraction is None:
            return 0.0, None

        return 1.0 - first_hit_fraction, first_hit_fraction

    def _spawn_row(self, *, y: float) -> None:
        """Spawn one logical gap row and two solid block sprites."""
        gap_width = self._rng.uniform(
            self.min_gap_width,
            self.max_gap_width,
        )
        half_gap = gap_width / 2.0
        min_center = self.edge_margin + half_gap
        max_center = self.width - self.edge_margin - half_gap
        gap_center = self._rng.uniform(min_center, max_center)

        row = GapRow(
            y=y,
            height=self.obstacle_height,
            speed=self.row_speed,
            gap_center=gap_center,
            gap_width=gap_width,
            world_width=self.width,
            world_height=self.height,
        )
        self.gap_rows.append(row)

        self.block_group.add(
            ObstacleBlockSprite(row=row, side="left"),
            ObstacleBlockSprite(row=row, side="right"),
        )

    def _remove_inactive_rows(self) -> None:
        """Remove logical rows that have moved outside the world."""
        self.gap_rows = [row for row in self.gap_rows if row.top < self.height + 80]

    def _has_collision(self) -> bool:
        """Return True if the player overlaps any solid block sprite."""
        return bool(
            pygame.sprite.spritecollide(
                self.player,
                self.block_group,
                dokill=False,
            )
        )

    def _row_passed(self, row: GapRow) -> bool:
        """Return True if a row passed the player."""
        return row.top > self.player.y + self.player.height / 2

    def _next_relevant_row(self) -> GapRow | None:
        """Return the closest row above or overlapping the player."""
        candidates = [row for row in self.gap_rows if row.bottom <= self.player.y]
        if not candidates:
            candidates = [row for row in self.gap_rows if row.top <= self.player.y]

        if not candidates:
            return None

        return max(candidates, key=lambda row: row.y)

    def _gap_alignment_score(self) -> float:
        """Return normalized horizontal alignment with the next gap."""
        row = self._next_relevant_row()
        if row is None:
            return 0.0

        distance = abs(self.player.x - row.gap_center)
        half_gap = row.gap_width / 2.0
        normalized_error = min(1.0, distance / max(1.0, half_gap))

        vertical_distance = max(0.0, self.player.y - row.bottom)
        vertical_weight = 1.0 - min(1.0, vertical_distance / self.player.y)

        return (1.0 - normalized_error) * vertical_weight

    def _is_near_wall(self) -> bool:
        """Return True if the player is close to a screen edge."""
        margin = self.player.width
        near_left = self.player.x < margin
        near_right = self.player.x > self.width - margin

        return near_left or near_right

    def _iter_blocks(self) -> Iterator[ObstacleBlockSprite]:
        """Yield active obstacle block sprites."""
        for sprite in self.block_group:
            if isinstance(sprite, ObstacleBlockSprite):
                yield sprite
