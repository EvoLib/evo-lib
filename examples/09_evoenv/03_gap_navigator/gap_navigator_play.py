# SPDX-License-Identifier: MIT
"""Play the GapNavigator environment manually."""

import pygame
from evoenv.cli import parse_env_args
from evoenv.core.difficulty import difficulty_task_path
from evoenv.core.env import Action, Observation
from evoenv.envs.gap_navigator_defaults import DEFAULT_FPS
from evoenv.envs.gap_navigator_task import GapNavigatorTask
from evoenv.renderers.pygame_gap_navigator import draw_env

args = parse_env_args(description="Play a GapNavigator agent.")
task = GapNavigatorTask.from_yaml(difficulty_task_path(args.difficulty))


class ManualGapNavigatorController:
    """
    Manual horizontal steering controller.

    Controls:
    - LEFT/A: move left
    - RIGHT/D: move right
    """

    def __init__(self) -> None:
        self.steering = 0.0

    def update(self) -> None:
        """Read keyboard state and update steering value."""
        keys = pygame.key.get_pressed()
        left = keys[pygame.K_LEFT] or keys[pygame.K_a]
        right = keys[pygame.K_RIGHT] or keys[pygame.K_d]

        if left and not right:
            self.steering = -1.0
        elif right and not left:
            self.steering = 1.0
        else:
            self.steering = 0.0

    def act(self, _observation: Observation) -> Action:
        """Return the current steering action."""
        return [self.steering]


def main() -> None:
    """Run the manual GapNavigator demo."""
    env = task.make_env()
    controller = ManualGapNavigatorController()

    pygame.init()
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("GapNavigator - Manual")
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

        controller.update()
        action = controller.act(observation)

        observation, _env_reward, done, info = env.step(action)
        total_reward += task.compute_reward(info)

        if done:
            print(f"Reward: {total_reward:.2f}")
            observation = env.reset()
            total_reward = 0.0

        draw_env(screen, env, total_reward, font, title="Manual GapNavigator")
        pygame.display.flip()
        clock.tick(DEFAULT_FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
