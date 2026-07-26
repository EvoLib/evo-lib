# SPDX-License-Identifier: MIT
"""Shared Pygame rendering helpers for EvoEnv debug episodes."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Generic, TypeVar

import pygame
from evoenv.core.controller import Controller
from evoenv.core.env import Env
from evoenv.core.sensors import SensorLineState
from PIL import Image

EnvT = TypeVar("EnvT", bound=Env)
DrawFunction = Callable[
    [pygame.Surface, EnvT, float, pygame.font.Font, str],
    None,
]


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
        if not self.enabled or step % self.frame_skip != 0:
            return

        width, height = surface.get_size()
        rgb_data = pygame.image.tostring(surface, "RGB")
        frame = Image.frombytes("RGB", (width, height), rgb_data)
        self.frames.append(frame)

    def save(self) -> Path | None:
        """Write all captured frames to disk and return the GIF path."""
        if self.filename is None or not self.frames:
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


class PygameDebugRenderer(Generic[EnvT]):
    """Render persistent Pygame debug episodes."""

    def __init__(
        self,
        *,
        caption: str,
        draw_env: DrawFunction[EnvT],
        fps: int,
        font_size: int = 24,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be greater than zero.")
        if font_size <= 0:
            raise ValueError("font_size must be greater than zero.")

        self.caption = caption
        self.draw_env = draw_env
        self.fps = int(fps)
        self.font_size = int(font_size)

        self.screen: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
        self.font: pygame.font.Font | None = None

    def run_episode(
        self,
        env: EnvT,
        controller: Controller,
        *,
        size: tuple[int, int],
        steps: int,
        seed: int | None,
        title: str,
        filename: str | Path | None = None,
        gif_fps: int,
        frame_skip: int = 1,
    ) -> Path | None:
        """Run one rendered episode and optionally write an animated GIF."""
        if steps < 0:
            raise ValueError("steps must not be negative.")

        screen, clock, font = self._get_resources(size)
        observation = env.reset(seed=seed)
        total_reward = 0.0
        recorder = GifRecorder(filename, fps=gif_fps, frame_skip=frame_skip)

        for step in range(steps):
            if self._should_stop():
                return recorder.save()

            action = controller.act(observation)
            observation, reward, done, _info = env.step(action)
            total_reward += reward

            self.draw_env(screen, env, total_reward, font, title)
            recorder.capture(screen, step=step)

            pygame.display.flip()
            clock.tick(self.fps)

            if done:
                break

        return recorder.save()

    def close(self) -> None:
        """Close Pygame and discard cached display resources."""
        pygame.quit()
        self.screen = None
        self.clock = None
        self.font = None

    def _get_resources(
        self,
        size: tuple[int, int],
    ) -> tuple[pygame.Surface, pygame.time.Clock, pygame.font.Font]:
        """Initialize and return persistent Pygame resources."""
        width, height = size
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be greater than zero.")

        if not pygame.get_init():
            pygame.init()
        if not pygame.display.get_init():
            pygame.display.init()
        if not pygame.font.get_init():
            pygame.font.init()

        display_surface = pygame.display.get_surface()
        if display_surface is None or display_surface.get_size() != size:
            self.screen = pygame.display.set_mode(size)
        else:
            self.screen = display_surface

        pygame.display.set_caption(self.caption)

        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.font is None:
            self.font = pygame.font.SysFont(None, self.font_size)

        return self.screen, self.clock, self.font

    def _should_stop(self) -> bool:
        """Return True when the current rendered episode should stop."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return True

            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return True

        return False


def draw_ray_sensors(
    screen: pygame.Surface,
    sensors: Iterable[SensorLineState],
) -> None:
    """Draw ray sensors using their current activation values."""
    for sensor in sensors:
        color = (255, 220, 80) if sensor.value > 0.0 else (90, 90, 90)

        pygame.draw.line(
            screen,
            color,
            (int(round(sensor.start_x)), int(round(sensor.start_y))),
            (int(round(sensor.end_x)), int(round(sensor.end_y))),
            2,
        )


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
