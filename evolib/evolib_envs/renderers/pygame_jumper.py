# SPDX-License-Identifier: MIT
"""Pygame rendering helpers for the Jumper environment."""

import pygame

from evolib.evolib_envs.core.controller import Controller
from evolib.evolib_envs.envs.jumper import JumperEnv
from evolib.evolib_envs.envs.jumper_defaults import (
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_MAX_STEPS,
    DEFAULT_WIDTH,
)
from evolib.evolib_envs.renderers.pygame_common import draw_text_overlay

FPS = DEFAULT_FPS


def draw_env(
    screen: pygame.Surface,
    env: JumperEnv,
    total_reward: float,
    font: pygame.font.Font,
    *,
    title: str = "Jumper",
) -> None:
    """Draw the full Jumper environment."""

    screen.fill((20, 20, 20))

    pygame.draw.line(
        screen,
        (180, 180, 180),
        (0, env.ground_y),
        (env.width, env.ground_y),
        3,
    )

    player_rect = pygame.Rect(0, 0, env.player.width, env.player.height)
    player_rect.centerx = int(round(env.player.x))
    player_rect.bottom = int(round(env.player.y))
    pygame.draw.rect(screen, (80, 180, 255), player_rect)

    obstacle_rect = pygame.Rect(0, 0, env.obstacle.width, env.obstacle.height)
    obstacle_rect.centerx = int(round(env.obstacle.x))
    obstacle_rect.bottom = int(round(env.obstacle.y))
    pygame.draw.rect(screen, (255, 120, 80), obstacle_rect)

    start, end = env.sensor.get_line(
        env.player.x,
        env.player.y,
        env.player.height,
    )

    pygame.draw.line(
        screen,
        (120, 120, 120),
        (int(start[0]), int(start[1])),
        (int(end[0]), int(end[1])),
        2,
    )

    lines = [
        title,
        f"reward={total_reward:.2f} step={env.step_count}",
        f"distance={env.obstacle.x - env.player.x:.1f} passed={env.passed_obstacles}",
        f"col={env.collision}",
        "ESC: quit | R: reset",
    ]

    draw_text_overlay(screen, font, lines)


class DebugRenderer:
    """Persistent Pygame renderer for debug episodes."""

    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((DEFAULT_WIDTH, DEFAULT_HEIGHT))
        pygame.display.set_caption("Jumper Debug")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)

    def run_episode(
        self,
        env: JumperEnv,
        controller: Controller,
        *,
        steps: int = DEFAULT_MAX_STEPS,
        seed: int | None,
        title: str,
    ) -> None:
        """Run one visual debug episode."""

        obs = env.reset(seed=seed)
        total_reward = 0.0

        for _ in range(steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return

            action = controller.act(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward

            draw_env(self.screen, env, total_reward, self.font, title=title)
            pygame.display.flip()
            self.clock.tick(FPS)

            if done:
                break


_DEBUG_RENDERER: DebugRenderer | None = None


def run_debug_episode(
    env: JumperEnv,
    controller: Controller,
    *,
    enabled: bool,
    generation: int,
    every: int = 5,
    steps: int = 500,
    seed: int | None = None,
    title: str = "Jumper Debug",
) -> None:
    """Run debug rendering periodically during training."""

    global _DEBUG_RENDERER

    if not enabled:
        return

    if generation % every != 0:
        return

    if _DEBUG_RENDERER is None:
        _DEBUG_RENDERER = DebugRenderer()

    _DEBUG_RENDERER.run_episode(
        env,
        controller,
        steps=steps,
        seed=seed,
        title=title,
    )
