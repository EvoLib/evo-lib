# SPDX-License-Identifier: MIT
"""Reusable simulation objects for the GapNavigator environment."""

from dataclasses import dataclass

import pygame


class AvoiderPlayer(pygame.sprite.Sprite):
    """Horizontally moving player body."""

    def __init__(
        self,
        *,
        x: float,
        y: float,
        speed: float,
        width: int = 32,
        height: int = 24,
    ) -> None:
        super().__init__()

        self.x = float(x)
        self.y = float(y)
        self.speed = float(speed)
        self.velocity_x = 0.0

        self.image = pygame.Surface((int(width), int(height)), pygame.SRCALPHA)
        self.rect = self.image.get_rect()
        self._sync_rect()

    def reset(self, *, x: float, y: float) -> None:
        """Reset the player position."""
        self.x = float(x)
        self.y = float(y)
        self.velocity_x = 0.0
        self._sync_rect()

    def step(self, steering: float, *, min_x: float, max_x: float) -> None:
        """Move horizontally using one clipped steering action."""
        clipped = max(-1.0, min(1.0, float(steering)))
        self.velocity_x = clipped * self.speed

        previous_x = self.x
        self.x += self.velocity_x
        self.x = max(min_x, min(max_x, self.x))

        if self.x != previous_x + self.velocity_x:
            self.velocity_x = self.x - previous_x

        self._sync_rect()

    @property
    def width(self) -> int:
        """Return the player width."""
        return int(self.rect.width)

    @property
    def height(self) -> int:
        """Return the player height."""
        return int(self.rect.height)

    def _sync_rect(self) -> None:
        """Synchronize the pygame rectangle with the float position."""
        self.rect.center = (int(round(self.x)), int(round(self.y)))


@dataclass
class GapRow:
    """Metadata for one logical obstacle row and its free gap."""

    y: float
    height: int
    speed: float
    gap_center: float
    gap_width: float
    world_width: int
    world_height: int
    counted: bool = False

    def step(self) -> None:
        """Move the logical row downward."""
        self.y += self.speed

    @property
    def gap_left(self) -> float:
        """Return the left x-coordinate of the gap."""
        return self.gap_center - self.gap_width / 2.0

    @property
    def gap_right(self) -> float:
        """Return the right x-coordinate of the gap."""
        return self.gap_center + self.gap_width / 2.0

    @property
    def top(self) -> float:
        """Return the top y-coordinate of the row."""
        return self.y - self.height / 2.0

    @property
    def bottom(self) -> float:
        """Return the bottom y-coordinate of the row."""
        return self.y + self.height / 2.0

    def gap_rect(self) -> pygame.Rect:
        """Return a rectangle describing the free gap area."""
        return pygame.Rect(
            int(round(self.gap_left)),
            int(round(self.top)),
            int(round(self.gap_width)),
            self.height,
        )


class ObstacleBlockSprite(pygame.sprite.Sprite):
    """One solid rectangular obstacle block belonging to a logical gap row."""

    def __init__(
        self,
        *,
        row: GapRow,
        side: str,
        color: tuple[int, int, int] = (255, 120, 80),
        offscreen_margin: int = 80,
    ) -> None:
        super().__init__()

        if side not in {"left", "right"}:
            raise ValueError("side must be 'left' or 'right'.")

        self.row = row
        self.side = side
        self.offscreen_margin = int(offscreen_margin)

        self.rect = self._make_initial_rect()

        self.image = pygame.Surface(
            (max(1, self.rect.width), max(1, self.rect.height)),
            pygame.SRCALPHA,
        )
        self.image.fill(color)

    def update(self) -> None:
        """Move the block rectangle with its logical row."""
        self.rect.y = int(round(self.row.top))

        if self.row.top >= self.row.world_height + self.offscreen_margin:
            self.kill()

    def _make_initial_rect(self) -> pygame.Rect:
        """Create the static block rectangle for this row side."""
        top = int(round(self.row.top))
        height = int(self.row.height)

        if self.side == "left":
            width = int(round(max(0.0, self.row.gap_left)))
            return pygame.Rect(0, top, width, height)

        x = int(round(min(float(self.row.world_width), self.row.gap_right)))
        width = self.row.world_width - x
        return pygame.Rect(x, top, width, height)
