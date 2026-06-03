# SPDX-License-Identifier: MIT
"""Run a simple sensor-based rule controller on GapNavigatorEnv."""

import pygame
from evoenv.cli import parse_env_args
from evoenv.core.controller import CallbackController
from evoenv.core.difficulty import difficulty_task_path
from evoenv.core.env import Action, Observation
from evoenv.envs.gap_navigator_defaults import DEFAULT_FPS
from evoenv.envs.gap_navigator_task import GapNavigatorTask
from evoenv.renderers.pygame_gap_navigator import draw_env

args = parse_env_args(description="Run a GapNavigator rule agent.")
task = GapNavigatorTask.from_yaml(difficulty_task_path(args.difficulty))


def gap_navigator_rule(observation: Observation) -> Action:
    """
    Steer away from the side with stronger obstacle sensor activation.

    The rule deliberately uses only sensor values and the agent's own velocity. It does
    not receive the gap center.
    """
    sensor_values = observation[:-2]
    velocity_x = observation[-1]

    if not sensor_values:
        return [0.0]

    midpoint = len(sensor_values) // 2
    left_pressure = sum(sensor_values[:midpoint])
    right_pressure = sum(sensor_values[midpoint:])
    steering = left_pressure - right_pressure

    steering -= velocity_x * 0.35

    return [max(-1.0, min(1.0, steering))]


def main() -> None:
    """Run the sensor-based GapNavigator demo."""
    env = task.make_env()
    controller = CallbackController(gap_navigator_rule)

    pygame.init()
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("EvoLib Env - GapNavigator Rule")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    observation = env.reset()
    total_reward = 0.0

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
        observation, _env_reward, done, info = env.step(action)
        total_reward += task.compute_reward(info)

        if done:
            print(f"Reward: {total_reward:.2f}")
            observation = env.reset()
            total_reward = 0.0

        draw_env(screen, env, total_reward, font, title="Sensor-rule GapNavigator")
        pygame.display.flip()
        clock.tick(DEFAULT_FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
