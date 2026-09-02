# SPDX-License-Identifier: MIT
"""Play the Collector environment manually."""

import pygame
from evoenv.core.env import Action, Observation
from evoenv.envs.collector_defaults import DEFAULT_FPS
from evoenv.envs.collector_task import CollectorTask
from evoenv.renderers.pygame_collector import draw_env
from evoenv.renderers.pygame_common import debug_display_size

TASK_CONFIG_PATH = "task.yaml"
FPS = DEFAULT_FPS


class ManualCollectorController:
    """
    Manual movement controller.

    Controls:
    - LEFT/RIGHT: turn
    - UP: move forward
    """

    def __init__(self) -> None:
        self.turn = 0.0
        self.throttle = 0.0

    def update(self) -> None:
        """Read keyboard state and update movement values."""
        keys = pygame.key.get_pressed()

        self.turn = 0.0
        if keys[pygame.K_LEFT]:
            self.turn -= 0.35
        if keys[pygame.K_RIGHT]:
            self.turn += 0.35

        self.throttle = 1.0 if keys[pygame.K_UP] else 0.0

    def act(self, _observation: Observation) -> Action:
        """Return the current movement action."""
        return [self.turn, self.throttle]


def main() -> None:
    """Run the manual Collector demo."""
    task = CollectorTask.from_yaml(TASK_CONFIG_PATH)
    env = task.make_env()
    controller = ManualCollectorController()

    pygame.init()
    screen = pygame.display.set_mode(debug_display_size((env.width, env.height)))
    pygame.display.set_caption("Collector - Manual")
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

        draw_env(screen, env, total_reward, font, title="Manual Collector")
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
