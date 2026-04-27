# SPDX-License-Identifier: MIT
"""Shared Pygame rendering for the LineFollower example."""

from __future__ import annotations

import math

import pygame

from evolib.evolib_envs.core.controller import Controller
from evolib.evolib_envs.envs.line_follower import LineFollowerEnv

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600
FPS = 60


def world_to_screen(x: float, y: float, env: LineFollowerEnv) -> tuple[int, int]:
    """Map world coordinates to screen coordinates."""

    screen_x = int((x + env.width * 0.5) / env.width * SCREEN_WIDTH)
    screen_y = int((env.height * 0.5 - y) / env.height * SCREEN_HEIGHT)
    return screen_x, screen_y


def draw_env(
    screen: pygame.Surface,
    env: LineFollowerEnv,
    total_reward: float,
    font: pygame.font.Font,
    *,
    title: str = "LineFollower",
) -> None:
    """Draw the current environment state."""

    screen.fill((20, 20, 20))

    line_points: list[tuple[int, int]] = []
    num_points = 240

    for i in range(num_points):
        x = -env.width * 0.5 + i * env.width / (num_points - 1)
        y = env.line_y(x)
        line_points.append(world_to_screen(x, y, env))

    pygame.draw.lines(screen, (220, 220, 220), False, line_points, 4)

    robot_x, robot_y = world_to_screen(env.x, env.y, env)

    pygame.draw.circle(screen, (80, 180, 255), (robot_x, robot_y), 16)

    nose_world_x = env.x + math.cos(env.angle) * 0.45
    nose_world_y = env.y + math.sin(env.angle) * 0.45
    nose_x, nose_y = world_to_screen(nose_world_x, nose_world_y, env)

    pygame.draw.line(screen, (255, 255, 255), (robot_x, robot_y), (nose_x, nose_y), 3)

    left_sensor, right_sensor = env.get_sensor_states()

    for sensor in (left_sensor, right_sensor):
        sx, sy = world_to_screen(sensor.x, sensor.y, env)
        intensity = int(80 + sensor.value * 175)

        pygame.draw.line(screen, (100, 100, 100), (robot_x, robot_y), (sx, sy), 1)
        pygame.draw.circle(screen, (255, intensity, 60), (sx, sy), 7)

    overlay_lines = [
        title,
        f"x={env.x:.2f} y={env.y:.2f} angle={env.angle:.2f}",
        f"left_sensor={left_sensor.value:.2f} right_sensor={right_sensor.value:.2f}",
        f"reward={total_reward:.2f} step={env.step_count}",
        "ESC: quit | R: reset",
    ]

    y_offset = 18
    for line in overlay_lines:
        text = font.render(line, True, (240, 240, 240))
        screen.blit(text, (18, y_offset))
        y_offset += 24


class DebugRenderer:
    """Pygame debug renderer."""

    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Training Debug")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)

    def run_episode(
        self,
        env: LineFollowerEnv,
        controller: Controller,
        *,
        generation: int,
        every: int,
        steps: int,
        seed: int,
        title: str,
    ) -> None:
        if generation % every != 0:
            return

        obs = env.reset(seed=seed)
        total_reward = 0.0

        step = 0
        running = True

        while running and step < steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            action = controller.act(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward

            draw_env(self.screen, env, total_reward, self.font, title=title)

            pygame.display.flip()
            self.clock.tick(FPS)

            if done:
                break

            step += 1


_DEBUG_RENDERER: DebugRenderer | None = None


def run_debug_episode(
    env: LineFollowerEnv,
    controller: Controller,
    *,
    enabled: bool,
    generation: int,
    every: int = 5,
    steps: int = 300,
    seed: int = 42,
    title: str = "Training Debug",
) -> None:
    """Run a debug episode using a persistent renderer."""

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
        every=every,
        generation=generation,
        seed=seed,
        title=title,
    )
