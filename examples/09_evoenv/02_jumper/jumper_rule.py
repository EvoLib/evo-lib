# SPDX-License-Identifier: MIT
"""Run a simple sensor-based rule controller on JumperEnv."""

import pygame
from evoenv.core.controller import CallbackController
from evoenv.core.env import Action, Observation
from evoenv.envs.jumper import JumperEnv
from evoenv.envs.jumper_defaults import (
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_MAX_STEPS,
    DEFAULT_WIDTH,
)
from evoenv.renderers.pygame_jumper import draw_env

SCREEN_WIDTH = DEFAULT_WIDTH
SCREEN_HEIGHT = DEFAULT_HEIGHT
MAX_STEPS = DEFAULT_MAX_STEPS
FPS = DEFAULT_FPS


def jumper_rule(observation: Observation) -> Action:
    """Jump when the sensor reports a close obstacle."""
    sensor_value = observation[0]
    normalized_obstacle_height = observation[1]

    should_jump = 0.58 <= sensor_value <= 0.92
    jump_strength = 0.65 + 0.35 * normalized_obstacle_height

    if should_jump:
        return [1.0, jump_strength]

    return [0.0, 0.0]


def main() -> None:
    """Run the rule-based Jumper demo."""
    env = JumperEnv(
        width=SCREEN_WIDTH,
        height=SCREEN_HEIGHT,
        max_steps=MAX_STEPS,
    )
    controller = CallbackController(jumper_rule)

    observation = env.reset()
    total_reward = 0.0

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("EvoEnv - Jumper Rule")
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
        observation, reward, done, _info = env.step(action)
        total_reward += reward

        if done:
            print(f"Reward: {total_reward:.2f}")
            observation = env.reset()
            total_reward = 0.0

        draw_env(screen, env, total_reward, font, title="Sensor-rule Jumper")
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
