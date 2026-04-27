# SPDX-License-Identifier: MIT
"""Watch a trained LineFollower individual with Pygame visualization."""

import sys

import pygame
from render import FPS, SCREEN_HEIGHT, SCREEN_WIDTH, draw_env

from evolib import Indiv, load_best_indiv
from evolib.evolib_envs.core.env import Action, Observation
from evolib.evolib_envs.envs.line_follower import LineFollowerEnv

RUN_NAME = "line_follower"

WORLD_WIDTH = 10.0
WORLD_HEIGHT = 6.0
MAX_STEPS = 500
SEED = 42


class LineFollowerController:
    """Minimal controller: directly forward observation through EvoNet."""

    def __init__(self, indiv: Indiv) -> None:
        self.net = indiv.para["brain"]

    def act(self, observation: Observation) -> Action:
        output = self.net.calc(observation)
        turn = max(-1.0, min(1.0, float(output[0])))
        return [turn]


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

        draw_env(screen, env, total_reward, font, title="Evolved LineFollower")

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
