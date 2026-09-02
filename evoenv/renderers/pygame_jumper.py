# SPDX-License-Identifier: MIT
"""Pygame rendering helpers for the Jumper environment."""

from pathlib import Path

import pygame
from evoenv.core.controller import Controller
from evoenv.envs.jumper import JumperEnv
from evoenv.envs.jumper_defaults import (
    DEFAULT_DEBUG_EVERY_N_GENERATIONS,
    DEFAULT_FPS,
)
from evoenv.renderers.pygame_common import (
    PygameDebugRenderer,
    draw_ray_sensors,
    draw_text_panel,
    split_debug_screen,
)


def draw_world(screen: pygame.Surface, env: JumperEnv) -> None:
    """Draw the Jumper world without debug text."""
    screen.fill((20, 20, 20))

    pygame.draw.line(
        screen,
        (90, 90, 90),
        (0, int(round(env.ground_y))),
        (env.width, int(round(env.ground_y))),
        2,
    )

    env.obstacle_group.draw(screen)
    draw_ray_sensors(screen, env.get_sensor_states())
    pygame.draw.rect(screen, (80, 180, 255), env.player.rect)


def draw_info(
    screen: pygame.Surface,
    env: JumperEnv,
    total_reward: float,
    font: pygame.font.Font,
    *,
    title: str,
) -> None:
    """Draw textual debug information in the side panel."""
    lines = [
        title,
        f"reward={total_reward:.2f}",
        f"step={env.step_count}",
        f"passed={env.passed_obstacles}",
        f"collisions={env.collision_count}",
        "ESC: quit",
    ]
    draw_text_panel(screen, font, lines)


def draw_env(
    screen: pygame.Surface,
    env: JumperEnv,
    total_reward: float,
    font: pygame.font.Font,
    title: str = "Jumper",
) -> None:
    """Draw the full Jumper debug frame."""
    world_screen, info_screen = split_debug_screen(
        screen,
        (env.width, env.height),
    )
    draw_world(world_screen, env)
    draw_info(
        info_screen,
        env,
        total_reward,
        font,
        title=title,
    )


_DEBUG_RENDERER = PygameDebugRenderer[JumperEnv](
    caption="Jumper Debug",
    draw_env=draw_env,
    fps=DEFAULT_FPS,
)


def run_debug_episode(
    env: JumperEnv,
    controller: Controller,
    *,
    enabled: bool,
    generation: int,
    every: int = DEFAULT_DEBUG_EVERY_N_GENERATIONS,
    steps: int = 500,
    seed: int | None = None,
    title: str = "Jumper Debug",
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
