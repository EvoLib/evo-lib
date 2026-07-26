# SPDX-License-Identifier: MIT
"""Pygame rendering helpers for the GapNavigator environment."""

from pathlib import Path

import pygame
from evoenv.core.controller import Controller
from evoenv.envs.gap_navigator import GapNavigatorEnv
from evoenv.envs.gap_navigator_defaults import (
    DEFAULT_DEBUG_EVERY_N_GENERATIONS,
    DEFAULT_FPS,
)
from evoenv.renderers.pygame_common import (
    PygameDebugRenderer,
    draw_ray_sensors,
    draw_text_overlay,
)


def draw_env(
    screen: pygame.Surface,
    env: GapNavigatorEnv,
    total_reward: float,
    font: pygame.font.Font,
    title: str = "GapNavigator",
) -> None:
    """Draw the full GapNavigator environment."""
    screen.fill((20, 20, 20))

    env.block_group.draw(screen)

    for row in env.gap_rows:
        pygame.draw.rect(screen, (40, 70, 45), row.gap_rect(), 1)

    draw_ray_sensors(screen, env.get_sensor_states())
    pygame.draw.rect(screen, (80, 180, 255), env.player.rect)

    lines = [
        title,
        f"reward={total_reward:.2f} step={env.step_count}",
        f"passed={env.passed_rows} collision={env.collision}",
        "ESC: quit",
    ]
    draw_text_overlay(screen, font, lines)


_DEBUG_RENDERER = PygameDebugRenderer[GapNavigatorEnv](
    caption="GapNavigator Debug",
    draw_env=draw_env,
    fps=DEFAULT_FPS,
)


def run_debug_episode(
    env: GapNavigatorEnv,
    controller: Controller,
    *,
    enabled: bool,
    generation: int,
    every: int = DEFAULT_DEBUG_EVERY_N_GENERATIONS,
    steps: int = 500,
    seed: int | None = None,
    title: str = "GapNavigator Debug",
    filename: str | Path | None = None,
    gif_fps: int = DEFAULT_FPS,
    frame_skip: int = 1,
) -> Path | None:
    """Run debug rendering periodically during training."""
    if not enabled or generation % every != 0:
        return None

    return _DEBUG_RENDERER.run_episode(
        env,
        controller,
        size=(env.width, env.height),
        steps=steps,
        seed=seed,
        title=title,
        filename=filename,
        gif_fps=gif_fps,
        frame_skip=frame_skip,
    )
