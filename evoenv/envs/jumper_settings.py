# SPDX-License-Identifier: MIT
"""Settings for the Jumper environment."""

from dataclasses import dataclass


@dataclass(frozen=True)
class JumperSettings:
    """Configuration bundle for the Jumper environment."""

    # Player physics
    gravity: float = 0.70
    jump_velocity: float = 15.5

    # Obstacle dynamics
    obstacle_speed: float = 5.0
    obstacle_width: int = 35
    min_obstacle_height: int = 25
    max_obstacle_height: int = 150
    min_spawn_gap: int = 250
    max_spawn_gap: int = 380

    # Reward / termination shaping
    terminate_on_collision: bool = False
    collision_penalty: float = 10.0
    pass_reward: float = 0.0  # 3.0
    alive_reward: float = 0.0  # 0.015
    jump_strength_penalty: float = 5.0


DEFAULT_JUMPER_SETTINGS = JumperSettings()
