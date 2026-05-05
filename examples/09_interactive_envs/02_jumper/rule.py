# SPDX-License-Identifier: MIT
"""Run a simple rule-based controller on JumperEnv."""

import sys

import pygame

from evolib.evolib_envs.cli import parse_jumper_args
from evolib.evolib_envs.core.env import Action, Observation
from evolib.evolib_envs.envs.jumper import JumperEnv
from evolib.evolib_envs.envs.jumper_defaults import (
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_MAX_STEPS,
    DEFAULT_WIDTH,
)
from evolib.evolib_envs.renderers.pygame_jumper import draw_env

SCREEN_WIDTH = DEFAULT_WIDTH
SCREEN_HEIGHT = DEFAULT_HEIGHT
MAX_STEPS = DEFAULT_MAX_STEPS
FPS = DEFAULT_FPS

args = parse_jumper_args()
difficulty = args.difficulty


class RuleBasedJumperController:
    """Jump when the obstacle is close and the player is on the ground."""

    def __init__(
        self, *, jump_distance_min: float = 0.10, jump_distance_max: float = 0.22
    ) -> None:
        self.jump_distance_min = jump_distance_min
        self.jump_distance_max = jump_distance_max

    def act(self, observation: Observation) -> Action:
        normalized_distance = observation[0]

        should_jump = (
            self.jump_distance_min <= normalized_distance <= self.jump_distance_max
        )
        return [1.0, 0.75 if should_jump else 0.0, 0.0]


def main() -> None:
    """Run the rule-based Jumper demo."""

    env = JumperEnv(
        width=SCREEN_WIDTH,
        height=SCREEN_HEIGHT,
        max_steps=MAX_STEPS,
        difficulty=difficulty,
    )
    controller = RuleBasedJumperController()

    observation = env.reset()
    total_reward = 0.0

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("EvoLib Env - Jumper Rule")
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

        draw_env(screen, env, total_reward, font, title="Rule-based Jumper")
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
