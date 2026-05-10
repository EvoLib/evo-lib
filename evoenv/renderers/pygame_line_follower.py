# SPDX-License-Identifier: MIT
"""Pygame rendering helpers for the LineFollower environment."""

import math
from pathlib import Path

import pygame
from evoenv.core.controller import Controller
from evoenv.envs.line_follower import LineFollowerEnv
from evoenv.envs.line_follower_defaults import (
    DEFAULT_DEBUG_EVERY_N_GENERATIONS,
    DEFAULT_DEBUG_MAX_STEPS,
    DEFAULT_FPS,
    DEFAULT_MAX_STEPS,
)
from evoenv.renderers.pygame_common import GifRecorder, draw_text_overlay


def draw_robot(screen: pygame.Surface, env: LineFollowerEnv) -> None:
    """Draw the robot body and heading direction."""

    robot = env.robot

    robot_x = int(round(robot.x))
    robot_y = int(round(robot.y))

    pygame.draw.circle(screen, (80, 180, 255), (robot_x, robot_y), robot.radius)

    nose_x = int(round(robot.x + math.cos(robot.angle) * 45.0))
    nose_y = int(round(robot.y + math.sin(robot.angle) * 45.0))

    pygame.draw.line(
        screen,
        (255, 255, 255),
        (robot_x, robot_y),
        (nose_x, nose_y),
        3,
    )


def draw_sensors(screen: pygame.Surface, env: LineFollowerEnv) -> None:
    """Draw robot sensors and their current contact state."""

    robot = env.robot
    robot_pos = (int(round(robot.x)), int(round(robot.y)))

    for sensor_state in env.get_sensor_states():
        sensor_pos = (
            int(round(sensor_state.x)),
            int(round(sensor_state.y)),
        )

        color = pygame.Color("green")
        if sensor_state.value != 0.0:
            color = pygame.Color("red")

        pygame.draw.line(screen, (100, 100, 100), robot_pos, sensor_pos, 1)
        pygame.draw.circle(screen, color, sensor_pos, robot.sensor_radius)


def draw_overlay(
    screen: pygame.Surface,
    env: LineFollowerEnv,
    total_reward: float,
    font: pygame.font.Font,
    *,
    title: str,
) -> None:
    """Draw textual debug information."""

    sensor_states = env.get_sensor_states()
    values = " ".join(f"{s.value:.0f}" for s in sensor_states)

    lines = [
        title,
        f"x={env.robot.x:.1f} y={env.robot.y:.1f} angle={env.robot.angle:.2f}",
        f"sensors=[{values}]",
        f"missed_line_steps={env.missed_line_steps}",
        f"reward={total_reward:.2f} step={env.step_count}",
        "ESC: quit",
    ]

    draw_text_overlay(screen, font, lines)


def draw_env(
    screen: pygame.Surface,
    env: LineFollowerEnv,
    total_reward: float,
    font: pygame.font.Font,
    *,
    title: str = "LineFollower",
) -> None:
    """Draw the full LineFollower environment."""

    screen.fill((20, 20, 20))
    screen.blit(env.line_surface, (0, 0))

    draw_robot(screen, env)
    draw_sensors(screen, env)
    draw_overlay(screen, env, total_reward, font, title=title)


class DebugRenderer:
    """Persistent Pygame renderer for debug episodes."""

    def __init__(self, *, width: int, height: int) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Training Debug")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)

    def run_episode(
        self,
        env: LineFollowerEnv,
        controller: Controller,
        *,
        steps: int = DEFAULT_MAX_STEPS,
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

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return gif_recorder.save()

            action = controller.act(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward

            draw_env(self.screen, env, total_reward, self.font, title=title)

            gif_recorder.capture(self.screen, step=step)
            pygame.display.flip()
            self.clock.tick(DEFAULT_FPS)

            if done:
                break

        return gif_recorder.save()


_DEBUG_RENDERER: DebugRenderer | None = None


def run_debug_episode(
    env: LineFollowerEnv,
    controller: Controller,
    *,
    enabled: bool,
    generation: int,
    every: int = DEFAULT_DEBUG_EVERY_N_GENERATIONS,
    steps: int = DEFAULT_DEBUG_MAX_STEPS,
    seed: int | None = None,
    title: str = "Training Debug",
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
