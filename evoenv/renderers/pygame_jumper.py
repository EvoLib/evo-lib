# SPDX-License-Identifier: MIT
"""Pygame rendering helpers for the Jumper environment."""

from pathlib import Path

import pygame
from evoenv.core.controller import Controller
from evoenv.envs.jumper import JumperEnv
from evoenv.envs.jumper_defaults import DEFAULT_FPS
from evoenv.renderers.pygame_common import GifRecorder, draw_text_overlay

FPS = DEFAULT_FPS


def _sensor_color(value: float) -> tuple[int, int, int]:
    """Return a simple sensor color based on activation value."""
    if value > 0.0:
        return (255, 220, 80)

    return (90, 90, 90)


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
        (90, 90, 90),
        (0, int(round(env.ground_y))),
        (env.width, int(round(env.ground_y))),
        2,
    )

    env.obstacle_group.draw(screen)

    for sensor in env.get_sensor_states():
        pygame.draw.line(
            screen,
            _sensor_color(sensor.value),
            (int(round(sensor.start_x)), int(round(sensor.start_y))),
            (int(round(sensor.end_x)), int(round(sensor.end_y))),
            2,
        )

    pygame.draw.rect(screen, (80, 180, 255), env.player.rect)

    lines = [
        title,
        f"reward={total_reward:.2f} step={env.step_count}",
        f"passed={env.passed_obstacles} collision={env.collision_count}",
        "ESC: quit",
    ]

    draw_text_overlay(screen, font, lines)


class DebugRenderer:
    """Persistent Pygame renderer for debug episodes."""

    def __init__(self, *, width: int, height: int) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Jumper Debug")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)

    def run_episode(
        self,
        env: JumperEnv,
        controller: Controller,
        *,
        steps: int,
        seed: int | None,
        title: str,
        filename: str | Path | None = None,
        gif_fps: int = DEFAULT_FPS,
        frame_skip: int = 1,
    ) -> Path | None:
        """Run one visual debug episode."""
        obs = env.reset(seed=seed)
        total_reward = 0.0
        gif_recorder = GifRecorder(filename, fps=gif_fps, frame_skip=frame_skip)

        for step in range(steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return gif_recorder.save()

                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return gif_recorder.save()

            action = controller.act(obs)
            obs, reward, done, _info = env.step(action)
            total_reward += reward

            draw_env(self.screen, env, total_reward, self.font, title=title)
            gif_recorder.capture(self.screen, step=step)
            pygame.display.flip()
            self.clock.tick(FPS)

            if done:
                break

        return gif_recorder.save()


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
    filename: str | Path | None = None,
    gif_fps: int = DEFAULT_FPS,
    frame_skip: int = 1,
) -> Path | None:
    """Run debug rendering periodically during training."""
    global _DEBUG_RENDERER

    if not enabled:
        return None

    if generation % every != 0:
        return None

    if _DEBUG_RENDERER is None:
        _DEBUG_RENDERER = DebugRenderer(width=env.width, height=env.height)

    return _DEBUG_RENDERER.run_episode(
        env,
        controller,
        steps=steps,
        seed=seed,
        title=title,
        filename=filename,
        gif_fps=gif_fps,
        frame_skip=frame_skip,
    )
