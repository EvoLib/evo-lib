# SPDX-License-Identifier: MIT
"""Play the Jumper environment manually."""

import pygame
from evoenv.core.env import Action, Observation
from evoenv.envs.jumper_defaults import DEFAULT_FPS
from evoenv.envs.jumper_task import JumperTask
from evoenv.renderers.pygame_jumper import draw_env

TASK_CONFIG_PATH = "task.yaml"
FPS = DEFAULT_FPS


class ManualJumperController:
    """
    Manual jump controller.

    Controls:
    - SPACE: jump
    """

    def __init__(self) -> None:
        self.jump = 0.0

    def update(self) -> None:
        """Read keyboard state and update jump value."""
        keys = pygame.key.get_pressed()
        self.jump = 1.0 if keys[pygame.K_SPACE] else 0.0

    def act(self, _observation: Observation) -> Action:
        """Return the current jump action."""
        return [self.jump, 1.0]


def main() -> None:
    """Run the manual Jumper demo."""
    task = JumperTask.from_yaml(TASK_CONFIG_PATH)
    env = task.make_env()
    controller = ManualJumperController()

    pygame.init()
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Jumper - Manual")
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

        observation, reward, done, _info = env.step(action)
        total_reward += reward

        if done:
            print(f"Reward: {total_reward:.2f}")
            observation = env.reset()
            total_reward = 0.0

        draw_env(screen, env, total_reward, font, title="Manual Jumper")
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
