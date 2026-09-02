# SPDX-License-Identifier: MIT
"""Pygame rendering helpers for the Collector environment."""

from __future__ import annotations

import math
from pathlib import Path

import pygame
from evoenv.core.controller import Controller
from evoenv.envs.collector import CollectorEnv
from evoenv.envs.collector_defaults import (
    DEFAULT_DEBUG_EVERY_N_GENERATIONS,
    DEFAULT_FPS,
)
from evoenv.renderers.pygame_common import (
    PygameDebugRenderer,
    draw_ray_sensors,
    draw_text_panel,
    split_debug_screen,
)


def draw_world(screen: pygame.Surface, env: CollectorEnv) -> None:
    """Draw the Collector world without debug text."""
    screen.fill((20, 20, 20))

    for obstacle in env.obstacles:
        pygame.draw.rect(screen, (90, 90, 90), obstacle.rect)

    for food in env.food_items:
        pygame.draw.circle(
            screen,
            (80, 220, 120),
            (int(round(food.x)), int(round(food.y))),
            food.radius,
        )

    draw_ray_sensors(screen, env.get_sensor_states())

    agent_pos = (int(round(env.agent.x)), int(round(env.agent.y)))
    pygame.draw.circle(screen, (80, 180, 255), agent_pos, env.agent.radius)

    heading_end = (
        int(round(env.agent.x + math.sin(env.agent.heading) * env.agent.radius * 2.0)),
        int(round(env.agent.y - math.cos(env.agent.heading) * env.agent.radius * 2.0)),
    )
    pygame.draw.line(screen, (180, 220, 255), agent_pos, heading_end, 2)


def draw_info(
    screen: pygame.Surface,
    env: CollectorEnv,
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
        f"food={env.food_collected} left={len(env.food_items)}",
        f"collisions={env.collision_count}",
        f"visited_cells={len(env.visited_cells)}",
        "ESC: quit",
    ]
    draw_text_panel(screen, font, lines)


def draw_env(
    screen: pygame.Surface,
    env: CollectorEnv,
    total_reward: float,
    font: pygame.font.Font,
    title: str = "Collector",
) -> None:
    """Draw the full Collector debug frame."""
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


_DEBUG_RENDERER = PygameDebugRenderer[CollectorEnv](
    caption="Collector Debug",
    draw_env=draw_env,
    fps=DEFAULT_FPS,
)


def run_debug_episode(
    env: CollectorEnv,
    controller: Controller,
    *,
    enabled: bool,
    generation: int,
    every: int = DEFAULT_DEBUG_EVERY_N_GENERATIONS,
    steps: int = 500,
    seed: int | None = None,
    title: str = "Collector Debug",
    filename: str | Path | None = None,
    gif_fps: int = DEFAULT_FPS,
    frame_skip: int = 1,
) -> Path | None:
    """Run debug rendering periodically during training."""
    if not enabled:
        return None

    if every <= 0:
        raise ValueError("every must be greater than zero.")

    if generation % every != 0:
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
