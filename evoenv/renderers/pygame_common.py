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

INFO_PANEL_WIDTH = 300


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


class PygameWindow:
    """Manage the Pygame window used by interactive EvoEnv examples."""

    def __init__(
        self,
        world_size: tuple[int, int],
        *,
        caption: str,
        fps: int,
        font_size: int = 24,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be greater than zero.")
        if font_size <= 0:
            raise ValueError("font_size must be greater than zero.")

        self.world_size = world_size
        self.fps = int(fps)

        if not pygame.get_init():
            pygame.init()
        if not pygame.display.get_init():
            pygame.display.init()
        if not pygame.font.get_init():
            pygame.font.init()

        self.screen = pygame.display.set_mode(debug_display_size(world_size))
        pygame.display.set_caption(caption)

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, font_size)
        self.running = True

    def process_events(self) -> bool:
        """Process window events and return True when a reset was requested."""
        reset_requested = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r:
                    reset_requested = True

        return reset_requested

    def update(self) -> None:
        """Present the current frame and limit the frame rate."""
        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self) -> None:
        """Close the Pygame window."""
        pygame.quit()
        self.running = False


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
        self.window: PygameWindow | None = None

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

        window = self._get_window(size)
        observation = env.reset(seed=seed)
        total_reward = 0.0
        recorder = GifRecorder(filename, fps=gif_fps, frame_skip=frame_skip)

        for step in range(steps):
            window.process_events()
            if not window.running:
                return recorder.save()

            action = controller.act(observation)
            observation, reward, done, _info = env.step(action)
            total_reward += reward

            self.draw_env(window.screen, env, total_reward, window.font, title)
            recorder.capture(window.screen, step=step)
            window.update()

            if done:
                break

        return recorder.save()

    def close(self) -> None:
        """Close Pygame and discard cached display resources."""
        if self.window is not None:
            self.window.close()
            self.window = None

    def _get_window(self, size: tuple[int, int]) -> PygameWindow:
        """Return a reusable Pygame window for the requested world size."""
        if (
            self.window is None
            or not self.window.running
            or self.window.world_size != size
        ):
            if self.window is not None:
                self.window.close()

            self.window = PygameWindow(
                size,
                caption=self.caption,
                fps=self.fps,
                font_size=self.font_size,
            )

        return self.window


def debug_display_size(size: tuple[int, int]) -> tuple[int, int]:
    """Return the full debug-window size for an environment world size."""
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be greater than zero.")

    return width + INFO_PANEL_WIDTH, height


def split_debug_screen(
    screen: pygame.Surface,
    world_size: tuple[int, int],
) -> tuple[pygame.Surface, pygame.Surface]:
    """Split a debug window into world and information surfaces."""
    width, height = world_size
    expected_size = debug_display_size(world_size)

    if screen.get_size() != expected_size:
        raise ValueError(
            f"debug screen must have size {expected_size}, got {screen.get_size()}."
        )

    world_screen = screen.subsurface(pygame.Rect(0, 0, width, height))
    info_screen = screen.subsurface(pygame.Rect(width, 0, INFO_PANEL_WIDTH, height))
    return world_screen, info_screen


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


def draw_text_panel(
    screen: pygame.Surface,
    font: pygame.font.Font,
    lines: list[str],
    *,
    background: tuple[int, int, int] = (28, 30, 36),
    border: tuple[int, int, int] = (70, 70, 76),
) -> None:
    """Draw debug information on a dedicated side panel."""
    screen.fill(background)
    pygame.draw.line(screen, border, (0, 0), (0, screen.get_height()), 1)
    draw_text_overlay(screen, font, lines)
