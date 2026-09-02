# SPDX-License-Identifier: MIT
"""Pygame rendering helpers for the LineFollower environment."""

import math
from pathlib import Path

import pygame
from evoenv.core.controller import Controller
from evoenv.core.sensors import SensorPointState
from evoenv.envs.line_follower import LineFollowerEnv
from evoenv.envs.line_follower_defaults import (
    DEFAULT_DEBUG_EVERY_N_GENERATIONS,
    DEFAULT_FPS,
)
from evoenv.renderers.pygame_common import (
    PygameDebugRenderer,
    draw_text_panel,
    split_debug_screen,
)


def draw_robot(screen: pygame.Surface, env: LineFollowerEnv) -> None:
    """Draw the robot body and heading direction."""
    robot = env.robot
    robot_pos = (int(round(robot.x)), int(round(robot.y)))

    pygame.draw.circle(screen, (80, 180, 255), robot_pos, robot.radius)

    nose_pos = (
        int(round(robot.x + math.cos(robot.angle) * 45.0)),
        int(round(robot.y + math.sin(robot.angle) * 45.0)),
    )
    pygame.draw.line(screen, (255, 255, 255), robot_pos, nose_pos, 3)


def draw_sensors(
    screen: pygame.Surface,
    env: LineFollowerEnv,
    sensor_states: list[SensorPointState],
) -> None:
    """Draw robot sensors and their current contact state."""
    robot = env.robot
    robot_pos = (int(round(robot.x)), int(round(robot.y)))

    for sensor_state in sensor_states:
        sensor_pos = (
            int(round(sensor_state.x)),
            int(round(sensor_state.y)),
        )
        color = pygame.Color("red" if sensor_state.value != 0.0 else "green")

        pygame.draw.line(screen, (100, 100, 100), robot_pos, sensor_pos, 1)
        pygame.draw.circle(screen, color, sensor_pos, robot.sensor_radius)


def draw_world(screen: pygame.Surface, env: LineFollowerEnv) -> None:
    """Draw the LineFollower world without debug text."""
    screen.fill((20, 20, 20))
    screen.blit(env.line_surface, (0, 0))

    sensor_states = env.get_sensor_states()
    draw_robot(screen, env)
    draw_sensors(screen, env, sensor_states)


def draw_info(
    screen: pygame.Surface,
    env: LineFollowerEnv,
    total_reward: float,
    font: pygame.font.Font,
    *,
    title: str,
) -> None:
    """Draw textual debug information in the side panel."""
    sensor_states = env.get_sensor_states()
    values = " ".join(f"{state.value:.0f}" for state in sensor_states)

    lines = [
        title,
        f"x={env.robot.x:.1f} y={env.robot.y:.1f}",
        f"angle={env.robot.angle:.2f}",
        f"sensors=[{values}]",
        f"missed_line_steps={env.missed_line_steps}",
        f"reward={total_reward:.2f}",
        f"step={env.step_count}",
        "ESC: quit",
    ]
    draw_text_panel(screen, font, lines)


def draw_env(
    screen: pygame.Surface,
    env: LineFollowerEnv,
    total_reward: float,
    font: pygame.font.Font,
    title: str = "LineFollower",
) -> None:
    """Draw the full LineFollower debug frame."""
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


_DEBUG_RENDERER = PygameDebugRenderer[LineFollowerEnv](
    caption="Training Debug",
    draw_env=draw_env,
    fps=DEFAULT_FPS,
)


def run_debug_episode(
    env: LineFollowerEnv,
    controller: Controller,
    *,
    enabled: bool,
    generation: int,
    every: int = DEFAULT_DEBUG_EVERY_N_GENERATIONS,
    steps: int | None = None,
    seed: int | None = None,
    title: str = "Training Debug",
    filename: str | Path | None = None,
    gif_fps: int = DEFAULT_FPS,
    frame_skip: int = 1,
) -> Path | None:
    """Run debug rendering periodically during training."""
    if not enabled or generation % every != 0:
        return None

    episode_steps = env.max_steps if steps is None else steps

    return _DEBUG_RENDERER.run_episode(
        env,
        controller,
        size=(env.width, env.height),
        steps=episode_steps,
        seed=seed,
        title=title,
        filename=filename,
        gif_fps=gif_fps,
        frame_skip=frame_skip,
    )
