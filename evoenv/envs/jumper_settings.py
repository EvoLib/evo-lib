# SPDX-License-Identifier: MIT
"""Difficulty presets for the Jumper environment."""

from dataclasses import dataclass

from evoenv.core.difficulty import Difficulty


@dataclass(frozen=True)
class JumperSettings:
    """Configuration bundle for one Jumper difficulty preset."""

    difficulty: Difficulty

    # Player physics
    gravity: float
    jump_velocity: float

    # Obstacle dynamics
    obstacle_speed: float
    obstacle_width: int
    obstacle_height: int
    min_spawn_gap: int
    max_spawn_gap: int

    # Reward / termination shaping
    terminate_on_collision: bool
    collision_penalty: float
    pass_reward: float
    alive_reward: float


JUMPER_SETTINGS: dict[Difficulty, JumperSettings] = {
    Difficulty.EASY: JumperSettings(
        difficulty=Difficulty.EASY,
        gravity=0.62,
        jump_velocity=12.0,
        obstacle_speed=4.0,
        obstacle_width=34,
        obstacle_height=38,
        min_spawn_gap=300,
        max_spawn_gap=430,
        terminate_on_collision=False,
        collision_penalty=10.0,
        pass_reward=3.0,
        alive_reward=0.015,
    ),
    Difficulty.MEDIUM: JumperSettings(
        difficulty=Difficulty.MEDIUM,
        gravity=0.70,
        jump_velocity=12.5,
        obstacle_speed=5.2,
        obstacle_width=38,
        obstacle_height=44,
        min_spawn_gap=250,
        max_spawn_gap=380,
        terminate_on_collision=False,
        collision_penalty=10.0,
        pass_reward=3.0,
        alive_reward=0.015,
    ),
    Difficulty.HARD: JumperSettings(
        difficulty=Difficulty.HARD,
        gravity=0.78,
        jump_velocity=13.0,
        obstacle_speed=6.4,
        obstacle_width=44,
        obstacle_height=52,
        min_spawn_gap=210,
        max_spawn_gap=330,
        terminate_on_collision=False,
        collision_penalty=10.0,
        pass_reward=3.0,
        alive_reward=0.015,
    ),
}


def get_jumper_settings(difficulty: str | Difficulty) -> JumperSettings:
    """Return Jumper settings for a difficulty name."""
    return JUMPER_SETTINGS[Difficulty(difficulty)]
