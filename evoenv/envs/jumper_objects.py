# SPDX-License-Identifier: MIT
"""Reusable simulation objects for the pixel-based Jumper environment."""

from dataclasses import dataclass

from evoenv.envs.jumper_defaults import DEFAULT_GROUND_Y


@dataclass
class JumperPlayer:
    """Simple player body with vertical jump physics."""

    x: float = 160.0
    y: float = float(DEFAULT_GROUND_Y)
    width: int = 30
    height: int = 42
    velocity_y: float = 0.0
    gravity: float = 0.75
    max_jump_velocity: float = 15.0
    ground_y: float = float(DEFAULT_GROUND_Y)

    def reset(self, *, ground_y: float) -> None:
        """Place the player on the ground and clear vertical velocity."""

        self.ground_y = float(ground_y)
        self.y = float(ground_y)
        self.velocity_y = 0.0

    @property
    def is_on_ground(self) -> bool:
        """Return True if the player is currently standing on the ground."""

        return self.velocity_y == 0.0 and self.y >= self.ground_y

    def jump(self, force: float = 1.0) -> None:
        """Apply a jump impulse."""

        force = max(0.0, min(1.0, force))
        self.velocity_y = -self.max_jump_velocity * force

    def step(self, *, ground_y: float) -> None:
        """Advance vertical physics by one simulation step."""

        self.ground_y = float(ground_y)
        self.y += self.velocity_y
        self.velocity_y += self.gravity

        if self.y >= ground_y:
            self.y = float(ground_y)
            self.velocity_y = 0.0


@dataclass
class JumperObstacle:
    """Moving obstacle in front of the player."""

    x: float
    y: float
    width: int = 32
    height: int = 42
    speed: float = 6.0

    def step(self) -> None:
        """Move the obstacle from right to left."""

        self.x -= self.speed


@dataclass
class JumperSensor:
    """Forward-facing distance sensor."""

    range: float

    def get_line(
        self, player_x: float, ground_y: float, player_height: float
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return sensor line in world coordinates."""

        y = ground_y - player_height / 2
        start = (player_x, y)
        end = (player_x + self.range, y)
        return start, end
