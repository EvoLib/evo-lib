# SPDX-License-Identifier: MIT
"""Watch a trained LineFollower individual with a simple Pygame visualization."""

from __future__ import annotations

import math
import sys

import pygame

from evolib import Indiv, load_best_indiv
from evolib.evolib_envs.core.env import Action, Observation
from evolib.evolib_envs.envs.line_follower import LineFollowerEnv

RUN_NAME = "line_follower"

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600
FPS = 60

WORLD_WIDTH = 10.0
WORLD_HEIGHT = 6.0
MAX_STEPS = 1400
SEED = 42


class LineFollowerController:
    """Minimal controller: directly forward observation through EvoNet."""

    def __init__(self, indiv: Indiv) -> None:
        self.net = indiv.para["brain"]

    def act(self, observation: Observation) -> Action:
        output = self.net.calc(observation)
        return [float(x) for x in output]


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
) -> None:
    """Draw the current environment state."""

    screen.fill((20, 20, 20))

    # Draw target line (curved)
    line_points: list[tuple[int, int]] = []

    num_points = 240

    for i in range(num_points):
        x = -env.width * 0.5 + i * env.width / (num_points - 1)
        y = env.line_y(x)

        sx, sy = world_to_screen(x, y, env)
        line_points.append((sx, sy))

    pygame.draw.lines(
        screen,
        (220, 220, 220),
        False,
        line_points,
        4,
    )

    # Draw robot body.
    robot_x, robot_y = world_to_screen(env.x, env.y, env)
    robot_radius = 16

    pygame.draw.circle(
        screen,
        (80, 180, 255),
        (robot_x, robot_y),
        robot_radius,
    )

    # Draw heading.
    nose_world_x = env.x + math.cos(env.angle) * 0.45
    nose_world_y = env.y + math.sin(env.angle) * 0.45
    nose_x, nose_y = world_to_screen(nose_world_x, nose_world_y, env)

    pygame.draw.line(
        screen,
        (255, 255, 255),
        (robot_x, robot_y),
        (nose_x, nose_y),
        3,
    )

    # Draw sensors from environment geometry.
    left_sensor, right_sensor = env.get_sensor_states()

    for sensor in (left_sensor, right_sensor):
        sx, sy = world_to_screen(sensor.x, sensor.y, env)

        intensity = int(80 + sensor.value * 175)
        color = (255, intensity, 60)

        pygame.draw.line(
            screen,
            (100, 100, 100),
            (robot_x, robot_y),
            (sx, sy),
            1,
        )

        pygame.draw.circle(
            screen,
            color,
            (sx, sy),
            7,
        )

    # Debug overlay.
    overlay_lines = [
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


def main() -> None:
    indiv = load_best_indiv(run_name=RUN_NAME)
    if indiv is None:
        raise RuntimeError(f"No best individual loaded for run_name={RUN_NAME!r}")

    env = LineFollowerEnv(
        width=WORLD_WIDTH,
        height=WORLD_HEIGHT,
        max_steps=MAX_STEPS,
    )
    controller = LineFollowerController(indiv)

    observation = env.reset(seed=SEED)
    total_reward = 0.0

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("EvoLib Env - LineFollower Watch")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                if event.key == pygame.K_r:
                    observation = env.reset(seed=SEED)
                    total_reward = 0.0

        action = controller.act(observation)
        observation, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            observation = env.reset(seed=SEED)
            total_reward = 0.0

        draw_env(screen, env, total_reward, font)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
