# SPDX-License-Identifier: MIT
"""
Pixel-based 2D LineFollower environment.

The environment uses Pygame-style pixel coordinates:
- origin is at the top-left
- x grows to the right
- y grows downward

The line is represented as a pygame Mask. Sensors are small circular masks and detect
whether they overlap the line mask.
"""

import math
import random

import pygame

from evolib.evolib_envs.core.env import Action, Env, Observation, StepResult
from evolib.evolib_envs.envs.line_follower_objects import (
    LineFollowerRobot,
    SensorState,
)


class LineFollowerEnv(Env):
    """Minimal pixel-based line follower environment with steering-only actions."""

    observation_size = 2
    action_size = 1

    def __init__(
        self,
        *,
        width: int = 1000,
        height: int = 600,
        max_steps: int = 1400,
        line_complexity: float = 2.5,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.max_steps = max_steps

        self.previous_x = 0.0
        self.step_count = 0
        self.missed_line_steps = 0
        self.max_missed_line_steps = 35

        self.line_complexity = line_complexity
        self.line_width = 6
        self.line_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self.line_mask = pygame.mask.Mask((self.width, self.height), fill=False)
        self.line_points = self._build_line_points()
        self._build_line_mask()

        self.robot = LineFollowerRobot()

    @property
    def x(self) -> float:
        """Return robot x-position."""

        return self.robot.x

    @property
    def y(self) -> float:
        """Return robot y-position."""

        return self.robot.y

    @property
    def angle(self) -> float:
        """Return robot angle."""

        return self.robot.angle

    @property
    def sensor_radius(self) -> int:
        """Return the robot sensor radius."""

        return self.robot.sensor_radius

    def reset(self, seed: int | None = None) -> Observation:
        """Reset the episode and return the initial observation."""

        if seed is not None:
            random.seed(seed)

        start_x, start_y = self.line_points[0]
        self.previous_x = float(start_x)
        self.step_count = 0
        self.missed_line_steps = 0

        self.robot.reset(
            x=float(start_x),
            y=float(start_y),
            angle=random.uniform(-0.20, 0.20),
        )

        return self._observe()

    def step(self, action: Action) -> StepResult:
        """Advance the simulation by one step."""

        turn = max(-1.0, min(1.0, float(action[0])))

        self.robot.step(turn)
        self.step_count += 1

        observation = self._observe()
        left_sensor = observation[0]
        right_sensor = observation[1]

        progress = max(0.0, self.robot.x - self.previous_x)
        self.previous_x = self.robot.x

        robot_on_line = self.robot.touches_line(self.line_mask)

        reward = 0.0
        if robot_on_line:
            reward += progress * 0.25
            self.missed_line_steps = 0
        else:
            reward -= 0.25
            self.missed_line_steps += 1

        done = (
            self.step_count >= self.max_steps
            or self.missed_line_steps >= self.max_missed_line_steps
            or self.robot.x >= self.width - self.line_width
            or self._is_out_of_bounds(self.robot.x, self.robot.y)
        )

        info = {
            "x": self.robot.x,
            "y": self.robot.y,
            "angle": self.robot.angle,
            "turn": turn,
            "left_sensor": left_sensor,
            "right_sensor": right_sensor,
            "missed_line_steps": self.missed_line_steps,
        }

        return observation, reward, done, info

    def get_sensor_states(self) -> tuple[SensorState, SensorState]:
        """Return current sensor states using the environment line mask."""

        return self.robot.get_sensor_states(self.line_mask)

    def _observe(self) -> Observation:
        sensor_states = self.robot.get_sensor_states(self.line_mask)
        return [sensor.value for sensor in sensor_states]

    def _is_out_of_bounds(self, x: float, y: float) -> bool:
        """Return True if a point is outside the environment."""

        return x < 0.0 or y < 0.0 or x >= self.width or y >= self.height

    def _build_line_points(self) -> list[tuple[int, int]]:
        """Build a line with increasing curvature over x."""

        points: list[tuple[int, int]] = []

        center_y = self.height * 0.5
        max_amplitude = self.height * 0.5

        for x in range(0, self.width, 10):
            progress = x / self.width

            # amplitude grows over x (start flat, later stronger curves)
            amplitude = max_amplitude * (progress**1.5)

            # sinus curve
            wave = math.sin(progress * 2 * math.pi * self.line_complexity)

            # linear downward drift
            drift = -self.height * 0.16 * progress

            y = center_y + wave * amplitude + drift

            points.append((x, int(y)))

        return points

    def _build_line_mask(self) -> None:
        """Draw the line into a surface and build the collision mask from it."""

        self.line_surface.fill((0, 0, 0, 0))
        pygame.draw.lines(
            self.line_surface,
            (255, 255, 255, 255),
            False,
            self.line_points,
            self.line_width,
        )
        self.line_mask = pygame.mask.from_surface(self.line_surface)
