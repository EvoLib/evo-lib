# SPDX-License-Identifier: MIT
"""Reusable simulation objects for the Jumper environment."""

from dataclasses import dataclass

import pygame


class JumperPlayer(pygame.sprite.Sprite):
    """Player body with vertical jump physics."""

    def __init__(
        self,
        *,
        x: float,
        ground_y: float,
        gravity: float,
        jump_velocity: float,
        width: int = 30,
        height: int = 42,
        color: tuple[int, int, int] = (80, 180, 255),
    ) -> None:
        super().__init__()

        self.x = float(x)
        self.ground_y = float(ground_y)
        self.gravity = float(gravity)
        self.jump_velocity = float(jump_velocity)
        self.velocity_y = 0.0

        self.image = pygame.Surface((int(width), int(height)), pygame.SRCALPHA)
        self.image.fill(color)
        self.rect = self.image.get_rect()

        self.y = self.ground_y - self.rect.height / 2.0
        self._sync_rect()

    @property
    def on_ground(self) -> bool:
        """Return True if the player is standing on the ground."""
        return self.rect.bottom >= int(round(self.ground_y)) and self.velocity_y >= 0.0

    @property
    def normalized_height(self) -> float:
        """Return normalized jump height above the ground."""
        height_above_ground = max(0.0, self.ground_y - float(self.rect.bottom))
        return min(1.0, height_above_ground / max(1.0, self.ground_y))

    def reset(self, *, x: float, ground_y: float) -> None:
        """Reset the player to the ground position."""
        self.x = float(x)
        self.ground_y = float(ground_y)
        self.y = self.ground_y - self.rect.height / 2.0
        self.velocity_y = 0.0
        self._sync_rect()

    def step(self, *, jump_signal: float, jump_force: float) -> None:
        """Advance jump physics by one simulation step."""
        clipped_signal = max(0.0, min(1.0, float(jump_signal)))
        clipped_force = max(0.0, min(1.0, float(jump_force)))

        if self.on_ground and clipped_signal > 0.5:
            force_scale = 0.65 + 0.45 * clipped_force
            self.velocity_y = -self.jump_velocity * force_scale

        self.velocity_y += self.gravity
        self.y += self.velocity_y

        floor_y = self.ground_y - self.rect.height / 2.0
        if self.y > floor_y:
            self.y = floor_y
            self.velocity_y = 0.0

        self._sync_rect()

    def _sync_rect(self) -> None:
        """Synchronize the pygame rectangle with the float position."""
        self.rect.center = (int(round(self.x)), int(round(self.y)))


@dataclass(eq=False)
class JumperObstacle(pygame.sprite.Sprite):
    """One rectangular obstacle moving from right to left."""

    x: float
    ground_y: float
    speed: float
    width: int
    height: int
    color: tuple[int, int, int] = (255, 120, 80)
    counted: bool = False

    def __post_init__(self) -> None:
        super().__init__()

        self.x = float(self.x)
        self.ground_y = float(self.ground_y)
        self.speed = float(self.speed)

        self.image = pygame.Surface(
            (int(self.width), int(self.height)), pygame.SRCALPHA
        )
        self.image.fill(self.color)
        self.rect = self.image.get_rect()
        self._sync_rect()

    def update(self) -> None:
        """Move the obstacle left by one simulation step."""
        self.x -= self.speed
        self._sync_rect()

        if self.rect.right < -80:
            self.kill()

    def _sync_rect(self) -> None:
        """Synchronize the pygame rectangle with the float position."""
        self.rect.midbottom = (int(round(self.x)), int(round(self.ground_y)))
