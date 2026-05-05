# SPDX-License-Identifier: MIT
"""
Minimal headless Jumper environment.

The task is intentionally simple:
- The player remains at a fixed x-position.
- Obstacles move from right to left.
- The controller decides when to jump.

This environment is designed as a second small EvoLib-Env example after
LineFollower. It tests timing-based control without introducing complex physics.
"""

import random

from evolib.evolib_envs.core.env import Action, Env, Observation, StepResult
from evolib.evolib_envs.envs.jumper_defaults import (
    DEFAULT_GROUND_Y,
    DEFAULT_HEIGHT,
    DEFAULT_MAX_STEPS,
    DEFAULT_WIDTH,
)
from evolib.evolib_envs.envs.jumper_objects import (
    JumperObstacle,
    JumperPlayer,
    JumperSensor,
)
from evolib.evolib_envs.envs.jumper_settings import (
    JumperDifficulty,
    JumperSettings,
    get_jumper_settings,
)


class JumperEnv(Env):
    """Small timing-based jumping environment."""

    action_size = 2

    def __init__(
        self,
        *,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        ground_y: int = DEFAULT_GROUND_Y,
        max_steps: int = DEFAULT_MAX_STEPS,
        difficulty: str | JumperDifficulty = JumperDifficulty.MEDIUM,
    ) -> None:
        self.settings: JumperSettings = get_jumper_settings(difficulty)

        self.observation_size = self.settings.observation_size
        self.action_size = 2

        self.width = int(width)
        self.height = int(height)
        self.ground_y = int(ground_y)
        self.max_steps = int(max_steps)

        self.player = JumperPlayer()
        self.obstacle = JumperObstacle(
            x=float(self.width + 200),
            y=float(self.ground_y),
        )
        self.step_count = 0
        self.passed_obstacles = 0
        self.collision = 0
        self._rng = random.Random()

        self.sensor = JumperSensor(range=self.settings.sensor_range)

    def reset(self, seed: int | None = None) -> Observation:
        """Reset the episode and return the initial observation."""

        if seed is not None:
            self._rng.seed(seed)

        self.step_count = 0
        self.passed_obstacles = 0
        self.collision = 0
        self.player.reset(ground_y=self.ground_y)
        self._reset_obstacle(initial=True)

        return self._observe()

    def step(self, action: Action) -> StepResult:
        """Advance the environment by one step."""

        reward = 0.00
        done = False

        jump_signal = action[0]
        jump_force = action[1]

        can_jump = self.player.is_on_ground or self.settings.allow_air_jump

        if jump_signal > 0.5 and can_jump:
            self.player.jump(force=jump_force)
            reward -= jump_force * 0.5

        self.player.step(ground_y=self.ground_y)
        self.obstacle.step()
        self.step_count += 1

        # Collisions do not end the episode.
        # Each overlapping frame is penalized to discourage clipping through obstacles.
        if self._has_collision():
            reward -= 5.0
            self.collision += 1

        if self._obstacle_passed():
            self.passed_obstacles += 1
            self._reset_obstacle(initial=False)

        done = (
            # self._has_collision()
            self.step_count
            >= self.max_steps
            # or self.player.y <= 0
        )

        info = {
            "player_y": self.player.y,
            "velocity_y": self.player.velocity_y,
            "obstacle_x": self.obstacle.x,
            "distance": self.obstacle.x - self.player.x,
            "passed_obstacles": self.passed_obstacles,
            "jump_force": jump_force,
            "collision": self.collision,
        }

        return self._observe(), reward, done, info

    def _observe(self) -> Observation:
        """Return normalized observation values."""

        distance = self.obstacle.x - self.player.x
        obstacle_visible = 1.0 if 0.0 <= distance <= self.settings.sensor_range else 0.0

        normalized_distance = max(
            0.0,
            min(1.0, distance / self.settings.sensor_range),
        )

        normalized_obstacle_height = max(
            0.0,
            min(1.0, self.obstacle.height / self.settings.max_obstacle_height),
        )

        normalized_player_height = max(
            0.0,
            min(1.0, (self.ground_y - self.player.y) / self.ground_y),
        )

        normalized_obstacle_width = max(
            0.0,
            min(1.0, self.obstacle.width / self.settings.max_obstacle_width),
        )

        if self.settings.difficulty == JumperDifficulty.EASY:
            return [normalized_distance, obstacle_visible]

        if self.settings.difficulty == JumperDifficulty.MEDIUM:
            return [
                normalized_distance,
                obstacle_visible,
                normalized_obstacle_height,
            ]

        return [
            normalized_distance,
            obstacle_visible,
            normalized_obstacle_height,
            normalized_player_height,
            normalized_obstacle_width,
        ]

    def _reset_obstacle(self, *, initial: bool) -> None:
        """Place the obstacle to the right of the screen."""

        base_x = self.width + (260 if initial else 80)
        jitter = self._rng.randint(0, 240)

        self.obstacle.x = float(base_x + jitter)
        self.obstacle.y = float(self.ground_y)

        if self.settings.variable_obstacle_height:
            self.obstacle.height = self._rng.randint(
                self.settings.min_obstacle_height,
                self.settings.max_obstacle_height,
            )
        else:
            self.obstacle.height = self.settings.max_obstacle_height

        if self.settings.variable_obstacle_width:
            self.obstacle.width = self._rng.randint(
                self.settings.min_obstacle_width,
                self.settings.max_obstacle_width,
            )
        else:
            self.obstacle.width = self.settings.max_obstacle_width

    def _obstacle_passed(self) -> bool:
        """Return True if the obstacle has moved behind the player."""

        return self.obstacle.x + self.obstacle.width < self.player.x

    def _has_collision(self) -> bool:
        """Return True if player and obstacle rectangles overlap."""

        player_left = self.player.x - self.player.width / 2
        player_right = self.player.x + self.player.width / 2
        player_top = self.player.y - self.player.height
        player_bottom = self.player.y

        obstacle_left = self.obstacle.x - self.obstacle.width / 2
        obstacle_right = self.obstacle.x + self.obstacle.width / 2
        obstacle_top = self.obstacle.y - self.obstacle.height
        obstacle_bottom = self.obstacle.y

        horizontal_overlap = (
            player_left < obstacle_right and player_right > obstacle_left
        )
        vertical_overlap = player_top < obstacle_bottom and player_bottom > obstacle_top

        return horizontal_overlap and vertical_overlap
