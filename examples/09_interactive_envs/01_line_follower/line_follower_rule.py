# SPDX-License-Identifier: MIT
"""Run a simple rule-based steering controller on LineFollowerEnv."""

import sys

import pygame
from evoenv.cli import parse_env_args
from evoenv.core.controller import CallbackController
from evoenv.core.env import Action, Observation
from evoenv.envs.line_follower import LineFollowerEnv
from evoenv.envs.line_follower_defaults import (
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_MAX_STEPS,
    DEFAULT_WIDTH,
)
from evoenv.renderers.pygame_line_follower import draw_env

SCREEN_WIDTH = DEFAULT_WIDTH
SCREEN_HEIGHT = DEFAULT_HEIGHT
MAX_STEPS = DEFAULT_MAX_STEPS
FPS = DEFAULT_FPS


args = parse_env_args(description="Run a Line Follower agent.")
difficulty = args.difficulty


def linefollower_rule(observation: Observation) -> Action:
    left_sensor, right_sensor = observation

    error = right_sensor - left_sensor
    turn = error
    turn = max(-1.0, min(1.0, turn))

    return [turn]


def main() -> None:
    env = LineFollowerEnv(
        width=SCREEN_WIDTH,
        height=SCREEN_HEIGHT,
        max_steps=MAX_STEPS,
        difficulty=difficulty,
    )

    controller = CallbackController(linefollower_rule)

    observation = env.reset()
    total_reward = 0.0

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("EvoLib Env - LineFollower Rule")
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
                    observation = env.reset()
                    total_reward = 0.0

        action = controller.act(observation)
        observation, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            print(f"Reward: {total_reward:.2f}")
            observation = env.reset()
            total_reward = 0.0

        draw_env(screen, env, total_reward, font, title="Rule-based LineFollower")

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
