# SPDX-License-Identifier: MIT
"""
2D LineFollower environment.

The agent controls steering only. Forward motion is provided by the environment.
"""

import math
import random
from dataclasses import dataclass

from evolib.evolib_envs.core.env import Action, Env, Observation, StepResult


@dataclass
class SensorState:
    """World-space sensor state for rendering and debugging."""

    x: float
    y: float
    value: float


class LineFollowerEnv(Env):
    """Minimal 2D line follower environment with steering-only actions."""

    observation_size = 2
    action_size = 1

    def __init__(
        self,
        *,
        width: float = 10.0,
        height: float = 6.0,
        max_steps: int = 1400,
    ) -> None:
        self.width = width
        self.height = height
        self.max_steps = max_steps

        self.x = 0.0
        self.previous_x = 0.0
        self.y = 0.0
        self.angle = 0.0
        self.step_count = 0

        self.base_speed = 0.045
        self.turn_strength = 0.075

        self.sensor_forward = 0.55
        self.sensor_side = 0.30
        self.sensor_range = 0.45

    def line_y(self, x: float) -> float:
        """Return the target line y-position at world x."""
        return -self.height * 0.1 + math.sin(x * 1.5) * 0.8 + x * 0.2

    def reset(self, seed: int | None = None) -> Observation:
        """Reset the episode and return the initial observation."""

        if seed is not None:
            random.seed(seed)

        self.x = -self.width / 2.0
        self.previous_x = self.x
        self.y = self.line_y(self.x)
        self.angle = random.uniform(-0.25, 0.25)
        self.step_count = 0

        return self._observe()

    def step(self, action: Action) -> StepResult:
        """Advance the simulation by one step."""

        turn = max(-1.0, min(1.0, float(action[0])))

        self.angle += turn * self.turn_strength
        self.x += math.cos(self.angle) * self.base_speed
        self.y += math.sin(self.angle) * self.base_speed

        self.step_count += 1

        observation = self._observe()

        line_error = abs(self.y - self.line_y(self.x))
        heading_error = abs(math.sin(self.angle))

        progress = self.x - self.previous_x
        self.previous_x = self.x

        line_quality = max(0.0, 1.0 - line_error)

        reward = 0.0
        reward += progress * 20.0 * line_quality
        reward -= line_error * 3.0
        reward -= heading_error * 0.3
        reward -= abs(turn) * 0.03

        done = (
            self.step_count >= self.max_steps
            or line_error > 1.0
            or self.x > self.width * 0.5
        )

        info = {
            "x": self.x,
            "y": self.y,
            "angle": self.angle,
            "line_error": line_error,
            "heading_error": heading_error,
            "turn": turn,
            "left_sensor": observation[0],
            "right_sensor": observation[1],
        }

        return observation, reward, done, info

    def get_sensor_states(self) -> tuple[SensorState, SensorState]:
        """Return left and right sensor positions and values."""

        left_x, left_y = self._sensor_position(side=-1.0)
        right_x, right_y = self._sensor_position(side=1.0)

        left_value = self._line_sensor_value(left_x, left_y)
        right_value = self._line_sensor_value(right_x, right_y)

        return (
            SensorState(left_x, left_y, left_value),
            SensorState(right_x, right_y, right_value),
        )

    def _observe(self) -> Observation:
        left_sensor, right_sensor = self.get_sensor_states()
        return [left_sensor.value, right_sensor.value]

    def _sensor_position(self, *, side: float) -> tuple[float, float]:
        """Return one sensor position in world coordinates."""

        forward_x = math.cos(self.angle)
        forward_y = math.sin(self.angle)

        right_x = -math.sin(self.angle)
        right_y = math.cos(self.angle)

        sensor_x = (
            self.x + forward_x * self.sensor_forward + right_x * self.sensor_side * side
        )
        sensor_y = (
            self.y + forward_y * self.sensor_forward + right_y * self.sensor_side * side
        )

        return sensor_x, sensor_y

    def _line_sensor_value(self, sensor_x: float, sensor_y: float) -> float:
        """Return line intensity at a sensor position."""

        line_y = self.line_y(sensor_x)
        distance = abs(sensor_y - line_y)

        return max(0.0, 1.0 - distance / self.sensor_range)
