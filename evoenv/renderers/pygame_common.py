# SPDX-License-Identifier: MIT

from pathlib import Path

import pygame
from PIL import Image


class GifRecorder:
    """Collect rendered Pygame frames and write them as an animated GIF."""

    def __init__(
        self,
        filename: str | Path | None,
        *,
        fps: int,
        frame_skip: int = 1,
    ) -> None:
        self.filename = Path(filename) if filename is not None else None
        self.fps = int(fps)
        self.frame_skip = max(1, int(frame_skip))
        self.frames: list[Image.Image] = []

        if self.fps <= 0:
            raise ValueError("fps must be greater than zero.")

    @property
    def enabled(self) -> bool:
        """Return True if this recorder writes a GIF."""

        return self.filename is not None

    def capture(self, surface: pygame.Surface, *, step: int) -> None:
        """Capture one frame from a Pygame surface when recording is enabled."""

        if not self.enabled:
            return

        if step % self.frame_skip != 0:
            return

        width, height = surface.get_size()
        rgb_data = pygame.image.tostring(surface, "RGB")
        frame = Image.frombytes("RGB", (width, height), rgb_data)
        self.frames.append(frame)

    def save(self) -> Path | None:
        """Write all captured frames to disk and return the GIF path."""

        if not self.enabled:
            return None

        if self.filename is None:
            return None

        if not self.frames:
            return None

        self.filename.parent.mkdir(parents=True, exist_ok=True)

        first, *rest = self.frames
        duration_ms = int(round(1000 / self.fps))
        first.save(
            self.filename,
            save_all=True,
            append_images=rest,
            duration=duration_ms,
            loop=0,
            optimize=True,
        )

        return self.filename


def draw_text_overlay(
    screen: pygame.Surface,
    font: pygame.font.Font,
    lines: list[str],
    *,
    x: int = 18,
    y: int = 18,
    line_height: int = 24,
    color: tuple[int, int, int] = (240, 240, 240),
) -> None:
    """Draw a simple text overlay with one line per entry."""

    y_offset = y

    for line in lines:
        text = font.render(line, True, color)
        screen.blit(text, (x, y_offset))
        y_offset += line_height
