# SPDX-License-Identifier: MIT
"""Run a simple target-and-sensor rule controller on CollectorEnv."""

import pygame
from evoenv.core.controller import CallbackController
from evoenv.core.env import Action, Observation
from evoenv.envs.collector_defaults import DEFAULT_FPS
from evoenv.envs.collector_task import CollectorTask
from evoenv.renderers.pygame_collector import draw_env

TASK_CONFIG_PATH = "task.yaml"
FPS = DEFAULT_FPS


def collector_rule(observation: Observation) -> Action:
    """Steer toward food and avoid obstacles detected by the front rays."""
    target_angle_sin = observation[0]
    left_sensor = observation[3]
    center_sensor = observation[4]
    right_sensor = observation[5]

    turn = target_angle_sin
    throttle = 0.85

    if center_sensor > 0.25:
        turn = -1.0 if left_sensor < right_sensor else 1.0
        throttle = 0.35
    elif left_sensor > 0.25:
        turn = 1.0
        throttle = 0.55
    elif right_sensor > 0.25:
        turn = -1.0
        throttle = 0.55

    return [turn, throttle]


def main() -> None:
    """Run the rule-based Collector demo."""
    task = CollectorTask.from_yaml(TASK_CONFIG_PATH)
    env = task.make_env()
    controller = CallbackController(collector_rule)

    observation = env.reset()
    total_reward = 0.0

    pygame.init()
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("EvoEnv - Collector Rule")
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

        draw_env(screen, env, total_reward, font, title="Rule Collector")
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
